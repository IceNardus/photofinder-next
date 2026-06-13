//! SuperPoint ONNX model wrapper for keypoint and descriptor extraction

use std::sync::Mutex;
use image::{GenericImageView, GrayImage};
use image::imageops::{resize, FilterType};
use tracing::info;
use ort::session::Session;
use ort::value::Tensor;

use crate::core::features::select_top_keypoints;

const INPUT_SIZE: u32 = 256;
const DESCRIPTOR_DIM: usize = 256;
const MAX_KEYPOINTS: usize = 1024;

/// Raw SuperPoint output before processing
pub struct SuperPointOutput {
    /// Keypoints [N, 2] pixel coordinates
    pub keypoints: Vec<f32>,
    /// Confidence scores [N]
    pub scores: Vec<f32>,
    /// Descriptors [N, 256]
    pub descriptors: Vec<f32>,
    pub num_keypoints: usize,
}

/// SuperPoint feature extractor
pub struct SuperPoint {
    session: Mutex<Option<Session>>,
    input_name: String,
    output_names: Vec<String>,
}

// Safety: SuperPoint wraps Session in Mutex, which ensures thread-safe access
unsafe impl Send for SuperPoint {}
unsafe impl Sync for SuperPoint {}

impl SuperPoint {
    pub fn new(model_path: &str) -> Result<Self, String> {
        info!("Loading SuperPoint from: {}", model_path);
        let session = Session::builder()
            .expect("Failed to create session")
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load SuperPoint: {}", e))?;

        let inputs = session.inputs();
        let outputs = session.outputs();

        let input_name = inputs.first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());

        let output_names: Vec<String> = outputs.iter()
            .map(|o| o.name().to_string())
            .collect();

        info!("SuperPoint input: {}, outputs: {:?}", input_name, output_names);
        info!("SuperPoint loaded successfully");

        Ok(Self {
            session: Mutex::new(Some(session)),
            input_name,
            output_names,
        })
    }

    /// Extract features from grayscale image
    pub fn extract(&self, gray_image: &GrayImage) -> Result<SuperPointOutput, String> {
        let mut guard = self.session.lock().unwrap();
        let session = guard.as_mut().ok_or("No session")?;

        // Resize to INPUT_SIZE
        let resized = resize(
            gray_image,
            INPUT_SIZE,
            INPUT_SIZE,
            FilterType::Lanczos3,
        );

        // Preprocess: convert to CHW format, normalize to [0, 1]
        let input_data = self.preprocess(&resized);

        let shape = [1_i64, 1, INPUT_SIZE as i64, INPUT_SIZE as i64];
        let input = Tensor::from_array((shape, input_data))
            .map_err(|e| format!("Tensor error: {}", e))?;

        let outputs = session.run(ort::inputs![self.input_name.clone() => input])
            .map_err(|e| format!("Inference error: {}", e))?;

        // Debug: print output info
        for (i, (name, value)) in outputs.iter().enumerate() {
            eprintln!("[SuperPoint] Output {} ({}): shape={:?}", i, name, value.shape());
        }

        // Helper to extract tensor as Vec<f32> (handling i64 case)
        let extract_f32 = |output: &ort::value::Value| -> Result<Vec<f32>, String> {
            // Try f32 first - returns (&[f32], Shape) so we need to owned()
            if let Ok((_, tensor)) = output.try_extract_tensor::<f32>() {
                return Ok(tensor.to_vec());
            }
            // Fall back to i64 and convert
            if let Ok((_, tensor)) = output.try_extract_tensor::<i64>() {
                return Ok(tensor.iter().map(|&x| x as f32).collect());
            }
            Err("Cannot extract tensor as f32 or i64".to_string())
        };

        // Parse outputs based on actual model format
        // Model: keypoints [1, 1024, 2], descriptors [1, 1024], scores [1, 1024, 256]
        let keypoints_flat = extract_f32(&outputs[0])?;
        let scores_or_desc_flat = extract_f32(&outputs[1])?;
        let desc_or_score_flat = extract_f32(&outputs[2])?;

        // The model outputs are in a different format than expected:
        // - keypoints: [1, 1024, 2] -> [1024, 2] keypoint coordinates
        // - outputs[1]: [1, 1024] -> could be scores or something else
        // - outputs[2]: [1, 1024, 256] -> could be the actual descriptors

        // Find actual keypoints (non-zero in keypoints output)
        let num_detected = keypoints_flat.len() / 2;
        eprintln!("[SuperPoint] Raw keypoints count: {}", num_detected);

        // Use keypoints_flat as keypoints [N, 2]
        let mut keypoints = Vec::with_capacity(num_detected * 2);
        for i in 0..num_detected {
            keypoints.push(keypoints_flat[i * 2]);
            keypoints.push(keypoints_flat[i * 2 + 1]);
        }

        // The descriptors in this model are [1, 1024, 256] format
        // scores_or_desc_flat [1, 1024] might be keypoint scores
        // desc_or_score_flat [1, 1024, 256] is the actual descriptor data
        let actual_desc_len = desc_or_score_flat.len();
        let desc_per_keypoint = actual_desc_len / num_detected.max(1);

        let descriptors: Vec<f32>;
        let scores: Vec<f32>;

        if desc_per_keypoint == 256 && num_detected > 0 {
            // desc_or_score_flat is [N*256], reshape to [N, 256] then flatten back
            // We need [N, 256] descriptors
            let mut desc_matrix = Vec::with_capacity(num_detected * 256);
            for n in 0..num_detected {
                for d in 0..256 {
                    desc_matrix.push(desc_or_score_flat[n * 256 + d]);
                }
            }
            descriptors = desc_matrix;
            // scores_flat from outputs[1] [1, 1024] are the keypoint scores
            scores = scores_or_desc_flat[..num_detected.min(scores_or_desc_flat.len())].to_vec();
        } else {
            // Fallback: use outputs[1] as descriptors and outputs[2] as scores
            descriptors = desc_or_score_flat;
            scores = scores_or_desc_flat;
        }

        Ok(SuperPointOutput {
            keypoints,
            scores,
            descriptors,
            num_keypoints: num_detected,
        })
    }

    /// Extract and select top-k keypoints
    pub fn extract_with_selection(
        &self,
        gray_image: &GrayImage,
        max_keypoints: usize,
    ) -> Result<SuperPointOutput, String> {
        let raw = self.extract(gray_image)?;

        if raw.num_keypoints <= max_keypoints {
            return Ok(raw);
        }

        let (keypoints, scores, descriptors) = select_top_keypoints(
            &raw.keypoints,
            &raw.scores,
            &raw.descriptors,
            max_keypoints,
        );

        Ok(SuperPointOutput {
            keypoints,
            scores,
            descriptors,
            num_keypoints: max_keypoints,
        })
    }

    /// Extract from image file path
    pub fn extract_from_path(&self, image_path: &str) -> Result<SuperPointOutput, String> {
        let img = image::open(image_path)
            .map_err(|e| format!("Failed to open image: {}", e))?
            .to_rgb8();

        let gray: GrayImage = image::imageops::grayscale(&img);
        self.extract(&gray)
    }

    /// Extract from RGB image
    pub fn extract_from_rgb(&self, rgb: &image::RgbImage) -> Result<SuperPointOutput, String> {
        let gray: GrayImage = image::imageops::grayscale(rgb);
        self.extract(&gray)
    }

    /// Resize and extract (for different sized patches)
    pub fn extract_resize(&self, gray_image: &GrayImage, target_size: u32) -> Result<SuperPointOutput, String> {
        let resized = resize(
            gray_image,
            target_size,
            target_size,
            FilterType::Lanczos3,
        );
        self.extract(&resized)
    }

    fn preprocess(&self, img: &GrayImage) -> Vec<f32> {
        let mut input = Vec::with_capacity(INPUT_SIZE as usize * INPUT_SIZE as usize);

        // Grayscale CHW, normalize to [0, 1]
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = img.get_pixel(x, y).0[0] as f32 / 255.0;
                input.push(pixel);
            }
        }

        input
    }
}

impl Drop for SuperPoint {
    fn drop(&mut self) {
        info!("SuperPoint session dropped");
    }
}
