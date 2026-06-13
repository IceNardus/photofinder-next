//! ArcFace Embedding Extraction

use std::sync::Mutex;
use image::{GenericImageView, RgbImage};
use tracing::info;
use ort::session::Session;
use ort::value::Tensor;

const INPUT_SIZE: u32 = 112;
const EMBEDDING_DIM: usize = 512;

pub struct ArcFace {
    session: Mutex<Option<Session>>,
}

// Safety: ArcFace wraps Session in Mutex, which ensures thread-safe access
unsafe impl Send for ArcFace {}
unsafe impl Sync for ArcFace {}

impl ArcFace {
    pub fn new(model_path: &str) -> Result<Self, String> {
        info!("Loading ArcFace from: {}", model_path);
        let session = Session::builder()
            .expect("Failed to create session")
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ArcFace: {}", e))?;

        Ok(Self { session: Mutex::new(Some(session)) })
    }

    /// Extract embedding from aligned face (main + flip augmentation)
    pub fn extract(&self, aligned_face: &RgbImage) -> Result<Vec<f32>, String> {
        self.extract_with_flip(aligned_face, false)
    }

    /// Extract with flip augmentation option
    pub fn extract_with_flip(&self, aligned_face: &RgbImage, use_flip: bool) -> Result<Vec<f32>, String> {
        let mut guard = self.session.lock().unwrap();
        let session = guard.as_mut().ok_or("No session")?;

        let emb_main = self.extract_internal(session, aligned_face, false)?;

        if !use_flip {
            return Ok(emb_main);
        }

        // Create horizontally flipped version
        let flipped = flip_horizontal(aligned_face);
        let emb_flip = self.extract_internal(session, &flipped, true)?;

        // Average and normalize
        let mut combined = Vec::with_capacity(EMBEDDING_DIM);
        for i in 0..EMBEDDING_DIM {
            combined.push(emb_main[i] + emb_flip[i]);
        }

        // L2 normalize
        let mut norm = 0.0f32;
        for v in &combined {
            norm += v * v;
        }
        norm = norm.sqrt();
        if norm > 1e-6 {
            let scale = 1.0 / norm;
            for v in &mut combined {
                *v *= scale;
            }
        }

        Ok(combined)
    }

    fn extract_internal(&self, session: &mut Session, face: &RgbImage, is_flip: bool) -> Result<Vec<f32>, String> {
        // Prepare input: RGB to BGR (or try RGB based on model version)
        // w600k_r50 typically uses BGR in most implementations
        let mut input_data = Vec::with_capacity(3 * INPUT_SIZE as usize * INPUT_SIZE as usize);

        // BGR CHW format
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = if is_flip {
                    // For flipped image, read from opposite side
                    face.get_pixel(INPUT_SIZE - 1 - x, y)
                } else {
                    face.get_pixel(x, y)
                };
                input_data.push((pixel[2] as f32 - 127.5) / 128.0);
            }
        }
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = if is_flip {
                    face.get_pixel(INPUT_SIZE - 1 - x, y)
                } else {
                    face.get_pixel(x, y)
                };
                input_data.push((pixel[1] as f32 - 127.5) / 128.0);
            }
        }
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = if is_flip {
                    face.get_pixel(INPUT_SIZE - 1 - x, y)
                } else {
                    face.get_pixel(x, y)
                };
                input_data.push((pixel[0] as f32 - 127.5) / 128.0);
            }
        }

        let shape = [1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
        let input = Tensor::from_array((shape, input_data))
            .map_err(|e| format!("Tensor error: {}", e))?;

        let outputs = session.run(ort::inputs![input])
            .map_err(|e| format!("Inference error: {}", e))?;

        let embedding_data = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1;

        // L2 normalize
        let mut norm = 0.0f32;
        for v in embedding_data.iter() {
            norm += v * v;
        }
        norm = norm.sqrt();

        let normalized = if norm > 1e-6 {
            let scale = 1.0 / norm;
            embedding_data.iter().map(|v| v * scale).collect()
        } else {
            embedding_data.to_vec()
        };

        Ok(normalized)
    }

    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }
}

/// Create horizontally flipped version of face
fn flip_horizontal(face: &RgbImage) -> RgbImage {
    let mut flipped = RgbImage::new(INPUT_SIZE, INPUT_SIZE);
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let pixel = face.get_pixel(INPUT_SIZE - 1 - x, y);
            flipped.put_pixel(x, y, *pixel);
        }
    }
    flipped
}

/// Check if face texture is valid (reject pure-color blocks, blackboards, paintings)
fn is_valid_face_texture(face: &RgbImage) -> bool {
    let mut total_brightness = 0.0f32;
    let mut pixels = Vec::with_capacity(INPUT_SIZE as usize * INPUT_SIZE as usize);

    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let pixel = face.get_pixel(x, y);
            let gray = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
            total_brightness += gray;
            pixels.push(gray);
        }
    }

    let mean = total_brightness / (INPUT_SIZE as f32 * INPUT_SIZE as f32);

    // Calculate variance
    let variance: f32 = pixels.iter().map(|&g| (g - mean).powi(2)).sum::<f32>()
        / (INPUT_SIZE as f32 * INPUT_SIZE as f32);

    // Reject if variance too low (smooth blocks like blackboards) or too high (noise)
    const MIN_VARIANCE: f32 = 200.0;
    const MAX_VARIANCE: f32 = 6500.0;
    if variance < MIN_VARIANCE || variance > MAX_VARIANCE {
        return false;
    }
    true
}

/// FaceFeature output structure
#[derive(Debug, Clone)]
pub struct FaceFeature {
    pub bbox: [f32; 4],
    pub detector_score: f32,
    pub blur_score: f32,
    pub pose_score: f32,
    pub face_area_score: f32,
    pub quality: f32,
    pub embedding: Vec<f32>,
}

impl FaceFeature {
    pub fn new(
        bbox: [f32; 4],
        detector_score: f32,
        blur_score: f32,
        pose_score: f32,
        face_area_score: f32,
        quality: f32,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            bbox,
            detector_score,
            blur_score,
            pose_score,
            face_area_score,
            quality,
            embedding,
        }
    }
}