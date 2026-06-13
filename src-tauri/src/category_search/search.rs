//! Category-based search logic

use crate::category_search::embedder::MobileClipEmbedder;
use crate::category_search::index::{CategoryIndex, ObjectMetadata};
use crate::category_search::detector::YoloV8Detector;
use image::RgbImage;
use std::path::{Path, PathBuf};

/// Configuration for category search
#[derive(Debug, Clone)]
pub struct CategorySearchConfig {
    /// Path to YOLOv8 ONNX model
    pub yolo_model_path: String,
    /// Path to MobileCLIP ONNX model
    pub mobileclip_model_path: String,
    /// Minimum similarity threshold for results
    pub min_similarity: f32,
    /// Maximum number of results to return
    pub max_results: usize,
}

impl Default for CategorySearchConfig {
    fn default() -> Self {
        Self {
            yolo_model_path: String::new(),
            mobileclip_model_path: String::new(),
            min_similarity: 0.7,
            max_results: 50,
        }
    }
}

/// Search result with similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Object metadata
    pub object: ObjectMetadata,
    /// Cosine similarity to prototype
    pub similarity: f32,
}

/// Category-based search engine
pub struct CategorySearch {
    config: CategorySearchConfig,
    detector: YoloV8Detector,
    embedder: MobileClipEmbedder,
    index: CategoryIndex,
}

impl CategorySearch {
    /// Create a new category search engine
    pub fn new(
        config: CategorySearchConfig,
        db_path: &Path,
    ) -> Result<Self, String> {
        let detector = YoloV8Detector::new(&config.yolo_model_path)?;
        let embedder = MobileClipEmbedder::new(&config.mobileclip_model_path)?;
        let index = CategoryIndex::new(db_path)?;

        Ok(Self {
            config,
            detector,
            embedder,
            index,
        })
    }

    /// Load existing index
    #[allow(dead_code)]
    pub fn load(
        config: CategorySearchConfig,
        db_path: &Path,
    ) -> Result<Self, String> {
        let detector = YoloV8Detector::new(&config.yolo_model_path)?;
        let embedder = MobileClipEmbedder::new(&config.mobileclip_model_path)?;
        let index = CategoryIndex::load(db_path)?;

        Ok(Self {
            config,
            detector,
            embedder,
            index,
        })
    }

    /// Index a single image - detect objects and add to index
    pub fn index_image(
        &mut self,
        image_path: &str,
        image_id: i64,
    ) -> Result<usize, String> {
        // Load image
        let img = image::open(image_path)
            .map_err(|e| format!("Failed to open image: {}", e))?
            .to_rgb8();

        // Detect objects
        let detections = self.detector.detect(&img)?;

        let mut indexed_count = 0;

        for obj in detections {
            // Crop object
            let crop = obj.bbox.crop_image(&img);

            // Skip very small crops
            if crop.width() < 20 || crop.height() < 20 {
                continue;
            }

            // Generate embedding
            let embedding = self.embedder.embed_image(&crop)?;

            // Add to index
            self.index.add_object(
                image_id,
                image_path,
                &obj.class_name,
                (obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2),
                obj.confidence,
                &embedding,
            )?;

            indexed_count += 1;
        }

        Ok(indexed_count)
    }

    /// Generate a prototype from reference images
    /// This creates a "category" by averaging embeddings from detected objects
    pub fn create_prototype_from_images(
        &self,
        image_paths: &[String],
    ) -> Result<Vec<f32>, String> {
        let mut embeddings = Vec::new();

        for path in image_paths {
            // Load image
            let img = match image::open(path) {
                Ok(img) => img.to_rgb8(),
                Err(e) => {
                    eprintln!("Failed to open {}: {}", path, e);
                    continue;
                }
            };

            // Detect objects (use all detected objects)
            let detections = match self.detector.detect(&img) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Detection failed for {}: {}", path, e);
                    continue;
                }
            };

            // Embed each detected object
            for obj in detections {
                let crop = obj.bbox.crop_image(&img);

                // Skip very small crops
                if crop.width() < 20 || crop.height() < 20 {
                    continue;
                }

                match self.embedder.embed_image(&crop) {
                    Ok(embedding) => embeddings.push(embedding),
                    Err(e) => eprintln!("Embedding failed for crop in {}: {}", path, e),
                }
            }
        }

        if embeddings.is_empty() {
            return Err("No objects detected in reference images".to_string());
        }

        // Create prototype by averaging
        Ok(MobileClipEmbedder::create_prototype(&embeddings))
    }

    /// Search for similar objects using a prototype
    pub fn search(&self, prototype: &[f32]) -> Result<Vec<SearchResult>, String> {
        // Search index
        let results = self.index.search(prototype, self.config.max_results)?;

        // Filter by similarity threshold and convert
        let filtered: Vec<SearchResult> = results
            .into_iter()
            .filter(|(_, sim)| *sim >= self.config.min_similarity)
            .map(|(object, similarity)| SearchResult { object, similarity })
            .collect();

        Ok(filtered)
    }

    /// Index all images in a folder
    #[allow(dead_code)]
    pub fn index_folder(
        &mut self,
        folder_path: &str,
    ) -> Result<usize, String> {
        use walkdir::WalkDir;

        let mut total_indexed = 0;
        let mut image_id = 0i64;

        for entry in WalkDir::new(folder_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            // Skip non-image files
            if !is_image_file(path) {
                continue;
            }

            match self.index_image(path.to_string_lossy().as_ref(), image_id) {
                Ok(count) => {
                    if count > 0 {
                        println!("Indexed {} objects from {}", count, path.display());
                        total_indexed += count;
                    }
                    image_id += 1;
                }
                Err(e) => {
                    eprintln!("Failed to index {}: {}", path.display(), e);
                }
            }
        }

        Ok(total_indexed)
    }

    /// Get index size
    #[allow(dead_code)]
    pub fn index_size(&self) -> usize {
        self.index.len()
    }
}

/// Check if a file is an image based on extension
fn is_image_file(path: &Path) -> bool {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    matches!(ext.as_deref(),
        Some("jpg") | Some("jpeg") | Some("png") | Some("webp") |
        Some("gif") | Some("bmp") | Some("tiff") | Some("tif")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((MobileClipEmbedder::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((MobileClipEmbedder::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_prototype_creation() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let prototype = MobileClipEmbedder::create_prototype(&embeddings);

        // Should be normalized
        let norm: f32 = prototype.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}