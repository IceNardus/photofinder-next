//! MobileCLIP-S2 embedding generation

use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};

/// MobileCLIP-S2 embedder for generating image embeddings
pub struct MobileClipEmbedder {
    session: Arc<Mutex<Option<Session>>>,
    input_size: u32,
    embedding_dim: usize,
    input_name: String,
    output_name: String,
}

unsafe impl Send for MobileClipEmbedder {}
unsafe impl Sync for MobileClipEmbedder {}

impl MobileClipEmbedder {
    /// Create a new embedder from an ONNX model file
    pub fn new(model_path: &str) -> Result<Self, String> {
        let session = Session::builder()
            .expect("Failed to create session")
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load MobileCLIP model: {}", e))?;

        let inputs = session.inputs();
        let outputs = session.outputs();

        let input_name = inputs.first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());

        let output_name = outputs.first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "embedding".to_string());

        // MobileCLIP-S2: input 224x224, output 512-dim
        let (input_size, embedding_dim) = (224, 512);

        Ok(Self {
            session: Arc::new(Mutex::new(Some(session))),
            input_size,
            embedding_dim,
            input_name,
            output_name,
        })
    }

    /// Preprocess image for MobileCLIP
    fn preprocess(&self, img: &RgbImage) -> Vec<f32> {
        // Resize to model input size
        let resized = image::imageops::resize(
            img,
            self.input_size,
            self.input_size,
            FilterType::Triangle,
        );

        // Convert to CHW format and normalize to [0, 1]
        let mut input_data = vec![0.0f32; 3 * self.input_size as usize * self.input_size as usize];

        for y in 0..self.input_size as usize {
            for x in 0..self.input_size as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let idx = y * self.input_size as usize + x;
                input_data[idx] = pixel[0] as f32 / 255.0;
                input_data[idx + (self.input_size * self.input_size) as usize] = pixel[1] as f32 / 255.0;
                input_data[idx + 2 * (self.input_size * self.input_size) as usize] = pixel[2] as f32 / 255.0;
            }
        }

        input_data
    }

    /// Generate embedding for a single image
    pub fn embed_image(&self, img: &RgbImage) -> Result<Vec<f32>, String> {
        let mut guard = self.session.lock().unwrap();
        let session = guard.as_mut().ok_or("No session")?;

        let input_data = self.preprocess(img);

        // Create input tensor [1, 3, H, W]
        let shape = [1_i64, 3, self.input_size as i64, self.input_size as i64];
        let input = Tensor::from_array((shape, input_data))
            .map_err(|e| format!("Tensor error: {}", e))?;

        let outputs = session
            .run(ort::inputs![self.input_name.clone() => input])
            .map_err(|e| format!("MobileCLIP inference failed: {}", e))?;

        // Get output tensor
        let embedding_data = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract embedding: {}", e))?.1;

        // L2 normalize
        let norm: f32 = embedding_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            let mut normalized = embedding_data.to_vec();
            for v in &mut normalized {
                *v /= norm;
            }
            Ok(normalized)
        } else {
            Ok(embedding_data.to_vec())
        }
    }

    /// Generate embedding for multiple images (batched)
    #[allow(dead_code)]
    pub fn embed_images(&self, images: &[RgbImage]) -> Result<Vec<Vec<f32>>, String> {
        let mut embeddings = Vec::with_capacity(images.len());

        for img in images {
            embeddings.push(self.embed_image(img)?);
        }

        Ok(embeddings)
    }

    /// Compute cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Average multiple embeddings to create a prototype
    pub fn create_prototype(embeddings: &[Vec<f32>]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0; 512];
        }

        if embeddings.len() == 1 {
            return embeddings[0].clone();
        }

        let dim = embeddings[0].len();
        let mut prototype = vec![0.0f32; dim];

        for emb in embeddings {
            for (i, v) in emb.iter().enumerate() {
                prototype[i] += v;
            }
        }

        let n = embeddings.len() as f32;
        for v in &mut prototype {
            *v /= n;
        }

        // L2 normalize
        let norm: f32 = prototype.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut prototype {
                *v /= norm;
            }
        }

        prototype
    }
}