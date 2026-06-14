//! MobileCLIP Object Search Pipeline
//!
//! Based on the new architecture:
//! [Scanning] Image → ROI Extraction → MobileCLIP Embedding → HNSW Index
//! [Query]    Query → ROI Extraction → MobileCLIP Embedding → HNSW Top-N → LightGlue精排 → Fusion Score → Top-K

use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::{Result, anyhow};
use image::{GenericImageView, RgbImage, DynamicImage};
use tracing::{info, warn};
use std::collections::HashMap;

use crate::ai::lightglue::LightGlueMatcher;
use crate::ai::lightglue::superpoint::SuperPoint;
use crate::category_search::embedder::MobileClipEmbedder;
use crate::search::object_index::{ObjectHnswIndex, ObjectSearchResult as ObjectIndexResult};
use crate::core::index::hnsw::{HnswIndex, SearchResult};

use super::roi_extractor::Region;

/// Object search result with fusion scoring
#[derive(Debug, Clone)]
pub struct ObjectSearchResult {
    pub image_id: i64,
    pub image_path: String,
    /// MobileCLIP cosine similarity (coarse score)
    pub embedding_score: f32,
    /// LightGlue inlier ratio (fine score)
    pub inlier_ratio: f32,
    /// Number of LightGlue inliers
    pub inlier_count: usize,
    /// Number of matches
    pub match_count: usize,
    /// Bounding box overlap between query and matched ROI
    pub bbox_overlap: f32,
    /// Color similarity
    pub color_similarity: f32,
    /// Final fusion confidence
    pub confidence: f32,
    /// Bounding box of matched object in candidate image
    pub matched_bbox: [f32; 4],
}

impl ObjectSearchResult {
    /// Compute fusion confidence
    /// confidence = α * embedding_score + β * inlier_ratio + γ * color_similarity
    /// Default: α=0.3, β=0.5, γ=0.2 (prioritize local feature match for jewelry/small objects)
    pub fn compute_confidence(&self, alpha: f32, beta: f32, gamma: f32) -> f32 {
        alpha * self.embedding_score + beta * self.inlier_ratio + gamma * self.color_similarity
    }
}

/// Object search configuration
#[derive(Debug, Clone)]
pub struct ObjectSearchConfig {
    /// Number of top candidates from HNSW coarse search
    pub hnsw_top_k: usize,
    /// Minimum LightGlue matches to consider a match
    pub min_lightglue_matches: usize,
    /// RANSAC inlier threshold (pixels)
    pub ransac_threshold: f32,
    /// Fusion scoring weights
    pub fusion_alpha: f32,  // embedding weight
    pub fusion_beta: f32,   // inlier_ratio weight
    pub fusion_gamma: f32,  // color weight
    /// Minimum bbox overlap for spatial matching (0.0 to 1.0)
    pub min_bbox_overlap: f32,
    /// Minimum ROI size (pixels)
    pub min_roi_size: u32,
    /// Maximum number of ROIs per image
    pub max_rois_per_image: usize,
}

impl Default for ObjectSearchConfig {
    fn default() -> Self {
        Self {
            hnsw_top_k: 20,
            min_lightglue_matches: 10,
            ransac_threshold: 4.0,
            fusion_alpha: 0.3,
            fusion_beta: 0.5,
            fusion_gamma: 0.2,
            min_bbox_overlap: 0.2,
            min_roi_size: 64,
            max_rois_per_image: 10,
        }
    }
}

/// Object search engine using MobileCLIP + LightGlue fusion
pub struct ObjectSearch {
    mobileclip: Arc<MobileClipEmbedder>,
    superpoint: Arc<SuperPoint>,
    lightglue: Arc<LightGlueMatcher>,
    config: ObjectSearchConfig,
    object_index: Arc<Mutex<Option<ObjectHnswIndex>>>,
}

impl ObjectSearch {
    pub fn new(
        mobileclip_path: &str,
        superpoint_path: &str,
        lightglue_path: &str,
        config: ObjectSearchConfig,
    ) -> Result<Self> {
        let mobileclip = MobileClipEmbedder::new(mobileclip_path)
            .map_err(|e| anyhow!("Failed to load MobileCLIP: {}", e))?;
        let superpoint = SuperPoint::new(superpoint_path)
            .map_err(|e| anyhow!("Failed to load SuperPoint: {}", e))?;
        let lightglue = LightGlueMatcher::new(lightglue_path)
            .map_err(|e| anyhow!("Failed to load LightGlue: {}", e))?;

        Ok(Self {
            mobileclip: Arc::new(mobileclip),
            superpoint: Arc::new(superpoint),
            lightglue: Arc::new(lightglue),
            config,
            object_index: Arc::new(Mutex::new(None)),
        })
    }

    /// Initialize the object index for HNSW search
    pub fn init_index(&self, data_dir: &Path) -> Result<()> {
        let mut index = ObjectHnswIndex::new(data_dir)
            .map_err(|e| anyhow!("Failed to create object index: {}", e))?;

        // Try to load existing index from disk
        match index.load() {
            Ok(()) => {
                eprintln!("[ObjectSearch] Successfully loaded existing index from disk");
            }
            Err(e) => {
                eprintln!("[ObjectSearch] No existing index to load (or load failed): {}", e);
            }
        }

        let mut guard = self.object_index.lock().unwrap();
        *guard = Some(index);
        info!("[ObjectSearch] ObjectHnswIndex initialized");
        Ok(())
    }

    /// Add an object embedding to the HNSW index
    pub fn add_to_index(&self, image_id: i64, image_path: &str, region_type: &str, bbox: [f32; 4], embedding: &[f32]) -> Result<()> {
        let mut index_guard = self.object_index.lock().unwrap();
        let index = index_guard.as_mut()
            .ok_or_else(|| anyhow!("Object index not initialized"))?;

        index.add_object(
            image_id,
            image_path,
            region_type,
            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
            embedding,
        ).map_err(|e| anyhow!("Failed to add to index: {}", e))?;

        Ok(())
    }

    /// Save the HNSW index to disk (data is already in SQLite, this is for HNSW binary file if needed)
    pub fn save_index(&self) -> Result<()> {
        // Data is already persisted to SQLite via CategoryIndex.add_object()
        // HNSW is in-memory only, but can be reconstructed from SQLite on reload
        info!("[ObjectSearch] Index data is persisted to SQLite");
        Ok(())
    }

    /// Extract ROIs from an image using selective search
    pub fn extract_rois(&self, image_path: &str) -> Result<Vec<Roi>> {
        let img = image::open(image_path)?;
        self.extract_rois_from_image(&img)
    }

    /// Extract ROIs from an already-loaded image
    pub fn extract_rois_from_image(&self, img: &DynamicImage) -> Result<Vec<Roi>> {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        eprintln!("[ROI:extract_rois_from_image] source=original, dims={}x{}", width, height);

        // Use selective search to find object regions
        let regions = super::roi_extractor::selective_search(&rgb, self.config.max_rois_per_image);
        eprintln!("[ROI:extract_rois_from_image] selective_search returned {} regions", regions.len());

        let mut rois = Vec::new();
        for region in regions {
            // Filter by size
            if region.bbox[2] < self.config.min_roi_size as f32 ||
               region.bbox[3] < self.config.min_roi_size as f32 {
                continue;
            }

            // Crop ROI
            let x1 = region.bbox[0] as u32;
            let y1 = region.bbox[1] as u32;
            let x2 = (x1 + region.bbox[2] as u32).min(width);
            let y2 = (y1 + region.bbox[3] as u32).min(height);

            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            eprintln!("[ROI:extract_rois_from_image] cropping region: ({},{}) {}x{} on {}x{} image",
                x1, y1, x2-x1, y2-y1, width, height);

            let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);

            // Resize to 224x224 for MobileCLIP
            let resized = crop.resize_exact(
                224, 224,
                image::imageops::FilterType::Triangle,
            );

            // Compute color histogram
            let color_hist = Self::compute_color_histogram(&resized);

            rois.push(Roi {
                bbox: region.bbox,
                region_type: region.region_type,
                color_hist,
            });
        }

        info!("[ROI] Extracted {} ROIs from image", rois.len());
        Ok(rois)
    }

    /// Extract MobileCLIP embedding for a crop
    pub fn extract_embedding(&self, crop: &DynamicImage) -> Result<Vec<f32>> {
        let rgb = crop.to_rgb8();
        self.mobileclip.embed_image(&rgb)
            .map_err(|e| anyhow!("MobileCLIP embedding failed: {}", e))
    }

    /// Extract SuperPoint features from an image
    pub fn extract_superpoint(&self, image_path: &str) -> Result<SuperPointFeatures> {
        let img = image::open(image_path)?;
        let rgb = img.to_rgb8();
        self.extract_superpoint_from_rgb(&rgb)
    }

    pub fn extract_superpoint_from_rgb(&self, rgb: &RgbImage) -> Result<SuperPointFeatures> {
        let (width, height) = rgb.dimensions();

        // Convert to grayscale for SuperPoint
        let gray = image::imageops::grayscale(&DynamicImage::ImageRgb8(rgb.clone()));

        let sp_output = self.superpoint.extract(&gray)
            .map_err(|e| anyhow!("SuperPoint failed: {}", e))?;

        Ok(SuperPointFeatures {
            width,
            height,
            keypoints: sp_output.keypoints,
            descriptors: sp_output.descriptors,
        })
    }

    /// Match two images using LightGlue
    pub fn match_images(
        &self,
        query_img: &RgbImage,
        candidate_img: &RgbImage,
        query_bbox: [f32; 4],
        candidate_bbox: [f32; 4],
    ) -> Result<MatchResult> {
        let query_kps = self.extract_superpoint_from_rgb(query_img)?;
        let candidate_kps = self.extract_superpoint_from_rgb(candidate_img)?;

        // Compute bbox overlap
        let overlap = Self::compute_bbox_overlap(query_bbox, candidate_bbox);

        // Match with LightGlue
        let lightglue_result = self.lightglue.match_features(
            &query_kps.keypoints,
            &query_kps.descriptors,
            &candidate_kps.keypoints,
            &candidate_kps.descriptors,
            (query_kps.width, query_kps.height),
            (candidate_kps.width, candidate_kps.height),
        ).map_err(|e| anyhow!("LightGlue match failed: {}", e))?;

        let inlier_ratio = if lightglue_result.matches.len() > 0 {
            lightglue_result.num_inliers as f32 / lightglue_result.matches.len() as f32
        } else {
            0.0
        };

        Ok(MatchResult {
            num_inliers: lightglue_result.num_inliers,
            match_count: lightglue_result.matches.len(),
            inlier_ratio,
            bbox_overlap: overlap,
        })
    }

    fn compute_bbox_overlap(a: [f32; 4], b: [f32; 4]) -> f32 {
        // a = [x1, y1, w, h], b = [x1, y1, w, h]
        let a_x2 = a[0] + a[2];
        let a_y2 = a[1] + a[3];
        let b_x2 = b[0] + b[2];
        let b_y2 = b[1] + b[3];

        let x1 = a[0].max(b[0]);
        let y1 = a[1].max(b[1]);
        let x2 = a_x2.min(b_x2);
        let y2 = a_y2.min(b_y2);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = a[2] * a[3] + b[2] * b[3] - intersection;

        if union <= 0.0 {
            return 0.0;
        }

        intersection / union
    }

    fn compute_color_histogram(img: &DynamicImage) -> Vec<f32> {
        let rgb = img.to_rgb8();
        let mut hist_r = [0u32; 32];
        let mut hist_g = [0u32; 32];
        let mut hist_b = [0u32; 32];

        for pixel in rgb.pixels() {
            hist_r[(pixel[0] / 8) as usize] += 1;
            hist_g[(pixel[1] / 8) as usize] += 1;
            hist_b[(pixel[2] / 8) as usize] += 1;
        }

        let total = (rgb.width() * rgb.height()) as f32;
        let mut result = Vec::with_capacity(96);
        for h in hist_r.iter() {
            result.push(*h as f32 / total);
        }
        for h in hist_g.iter() {
            result.push(*h as f32 / total);
        }
        for h in hist_b.iter() {
            result.push(*h as f32 / total);
        }
        result
    }

    pub fn color_histogram_intersection(h1: &[f32], h2: &[f32]) -> f32 {
        h1.iter().zip(h2.iter())
            .map(|(a, b)| a.min(*b))
            .sum()
    }

    /// Search for similar objects: ROI extraction → MobileCLIP embedding → (HNSW or brute force) → LightGlue精排 → Fusion
    pub fn search(&self, query_image: &str, top_k: usize) -> Result<Vec<ObjectSearchResult>> {
        eprintln!("[ObjectSearch] search: query_image={}", query_image);

        // Step 0: Check if this is a crop query image (user-cropped)
        // For crop query images, we should use the ENTIRE image, not extract more ROIs
        let is_crop_query = query_image.contains("query_crop")
            || query_image.contains("crop_query")
            || query_image.contains("cropped");

        let query_img = image::open(query_image)?.to_rgb8();
        eprintln!("[ObjectSearch] query_img dimensions: {}x{}", query_img.width(), query_img.height());

        // Step 1: Extract ROIs from query image
        // IMPORTANT: For crop query images, skip ROI extraction and use full image
        let query_rois = if is_crop_query {
            eprintln!("[ObjectSearch] Detected crop query image - using full image (skipping ROI extraction)");
            // Return a single ROI covering the entire image
            vec![Roi {
                bbox: [0.0, 0.0, query_img.width() as f32, query_img.height() as f32],
                region_type: "full_crop_query",
                color_hist: vec![],
            }]
        } else {
            self.extract_rois(query_image)?
        };

        if query_rois.is_empty() {
            info!("[ObjectSearch] No ROIs extracted from query image");
            return Ok(vec![]);
        }
        eprintln!("[ObjectSearch] Using {} ROIs for query (first: {:?})", query_rois.len(), query_rois.first().map(|r| r.bbox));
        info!("[ObjectSearch] Extracted {} ROIs from query", query_rois.len());

        // Step 3: Create prototype from query ROIs using MobileCLIP
        let mut query_embeddings = Vec::new();
        let mut query_roi_with_emb = Vec::new();

        // Reuse the already-loaded query image instead of reloading for each ROI
        for roi in &query_rois {
            let x1 = roi.bbox[0] as u32;
            let y1 = roi.bbox[1] as u32;
            let x2 = (x1 + roi.bbox[2] as u32).min(query_img.width());
            let y2 = (y1 + roi.bbox[3] as u32).min(query_img.height());

            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            let crop = DynamicImage::ImageRgb8(query_img.clone()).crop_imm(x1, y1, x2 - x1, y2 - y1);
            match self.extract_embedding(&crop) {
                Ok(emb) => {
                    query_embeddings.push(emb);
                    query_roi_with_emb.push(roi.clone());
                }
                Err(e) => {
                    warn!("[ObjectSearch] Embedding failed: {}", e);
                }
            }
        }

        if query_embeddings.is_empty() {
            info!("[ObjectSearch] No embeddings generated from ROIs");
            return Ok(vec![]);
        }
        info!("[ObjectSearch] Generated {} embeddings", query_embeddings.len());

        // Create prototype by averaging embeddings
        let prototype = MobileClipEmbedder::create_prototype(&query_embeddings);

        // Log prototype embedding (first 10 values)
        eprintln!("[ObjectSearch] Query prototype embedding (first 10 of {} dims): {:?}", prototype.len(), &prototype[..10.min(prototype.len())]);
        eprintln!("[ObjectSearch] Query prototype stats: min={:.4}, max={:.4}, avg={:.4}",
            prototype.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            prototype.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            prototype.iter().sum::<f32>() / prototype.len() as f32
        );

        // Step 4: HNSW search using new ObjectHnswIndex
        let index_guard = self.object_index.lock().unwrap();
        let object_index = match index_guard.as_ref() {
            Some(idx) => idx,
            None => {
                info!("[ObjectSearch] Object index not initialized");
                return Ok(vec![]);
            }
        };

        // ObjectHnswIndex.search returns Top-N images (aggregated from objects)
        let coarse_results = object_index.search(&prototype, self.config.hnsw_top_k * 2);
        drop(index_guard);

        if coarse_results.is_empty() {
            eprintln!("[ObjectSearch] HNSW returned NO results - index might be empty!");
            return Ok(vec![]);
        }
        eprintln!("[ObjectSearch] HNSW returned {} image candidates:", coarse_results.len());
        for (i, cr) in coarse_results.iter().take(10).enumerate() {
            eprintln!("  [{}] image_id={}, score={:.4}, path={}", i, cr.image_id, cr.score, cr.image_path);
        }

        // Step 5: LightGlue fine ranking for each candidate image
        // Extract query features ONCE before loop (not per candidate)
        let query_kps = match self.extract_superpoint_from_rgb(&query_img) {
            Ok(kps) => {
                eprintln!("[ObjectSearch] Query SuperPoint extracted: {} keypoints", kps.keypoints.len());
                kps
            },
            Err(e) => {
                eprintln!("[ObjectSearch] FATAL: SuperPoint failed for query: {}", e);
                return Ok(vec![]);  // Return early - can't search without query features
            }
        };

        eprintln!("[ObjectSearch] Entering fine ranking loop, coarse_results.len={}, hnsw_top_k={}", coarse_results.len(), self.config.hnsw_top_k);
        let mut final_results = Vec::new();

        for (idx, coarse_result) in coarse_results.iter().take(self.config.hnsw_top_k).enumerate() {
            eprintln!("[ObjectSearch] Processing candidate {} of {}", idx, coarse_results.len().min(self.config.hnsw_top_k));
            // Skip if image doesn't exist
            if !std::path::Path::new(&coarse_result.image_path).exists() {
                eprintln!("[ObjectSearch] Candidate image not found: {}", coarse_result.image_path);
                continue;
            }

            // Load candidate image
            let candidate_img = match image::open(&coarse_result.image_path) {
                Ok(img) => img.to_rgb8(),
                Err(e) => {
                    warn!("[ObjectSearch] Failed to open candidate {}: {}", coarse_result.image_path, e);
                    continue;
                }
            };

            // Extract candidate features
            let candidate_kps = match self.extract_superpoint_from_rgb(&candidate_img) {
                Ok(kps) => {
                    if kps.keypoints.is_empty() {
                        eprintln!("[ObjectSearch] Candidate {} has 0 keypoints, skipping", coarse_result.image_path);
                        continue;
                    }
                    kps
                },
                Err(e) => {
                    warn!("[ObjectSearch] SuperPoint failed for candidate: {}", e);
                    continue;
                }
            };

            let lightglue_result = match self.lightglue.match_features(
                &query_kps.keypoints,
                &query_kps.descriptors,
                &candidate_kps.keypoints,
                &candidate_kps.descriptors,
                (query_kps.width, query_kps.height),
                (candidate_kps.width, candidate_kps.height),
            ) {
                Ok(r) => r,
                Err(e) => {
                    warn!("[ObjectSearch] LightGlue match failed: {}", e);
                    continue;
                }
            };

            eprintln!("[ObjectSearch] LightGlue: query_kps={}, cand_kps={}, matches={}, inliers={}",
                query_kps.keypoints.len(), candidate_kps.keypoints.len(),
                lightglue_result.matches.len(), lightglue_result.num_inliers);

            eprintln!("[ObjectSearch] >>> Added candidate to final_results, current len={}", final_results.len() + 1);

            let inlier_ratio = if lightglue_result.matches.len() > 0 {
                lightglue_result.num_inliers as f32 / lightglue_result.matches.len() as f32
            } else {
                0.0
            };

            // Compute bbox overlap between query image bounds and candidate ROI
            // For crop queries, query image IS the cropped region, so use full image bounds
            // For normal queries, use first ROI bbox as representative
            let query_bbox_for_overlap = if is_crop_query {
                [0.0, 0.0, query_img.width() as f32, query_img.height() as f32]
            } else if let Some(first_roi) = query_rois.first() {
                first_roi.bbox
            } else {
                [0.0, 0.0, query_img.width() as f32, query_img.height() as f32]
            };

            // Compute candidate bbox in [x, y, w, h] format
            let candidate_bbox = [coarse_result.bbox.0, coarse_result.bbox.1, coarse_result.bbox.2 - coarse_result.bbox.0, coarse_result.bbox.3 - coarse_result.bbox.1];
            let bbox_overlap = Self::compute_bbox_overlap(query_bbox_for_overlap, candidate_bbox);

            // Compute fusion confidence
            let confidence = self.config.fusion_alpha * coarse_result.score
                + self.config.fusion_beta * inlier_ratio
                + self.config.fusion_gamma * bbox_overlap;

            eprintln!("[ObjectSearch] Candidate result: embedding_score={:.4}, inlier_ratio={:.4}, bbox_overlap={:.4}, confidence={:.4}",
                coarse_result.score, inlier_ratio, bbox_overlap, confidence);

            final_results.push(ObjectSearchResult {
                image_id: coarse_result.image_id,
                image_path: coarse_result.image_path.clone(),
                embedding_score: coarse_result.score,
                inlier_ratio,
                inlier_count: lightglue_result.num_inliers,
                match_count: lightglue_result.matches.len(),
                bbox_overlap,
                color_similarity: bbox_overlap,
                confidence,
                matched_bbox: [coarse_result.bbox.0, coarse_result.bbox.1, coarse_result.bbox.2, coarse_result.bbox.3],
            });

            if final_results.len() >= top_k {
                break;
            }
        }

        // Sort by confidence and return top_k
        final_results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Filter: embedding_score (similarity) >= 0.5 and confidence >= 0.5
        let min_score = 0.5;
        final_results.retain(|r| r.embedding_score >= min_score && r.confidence >= min_score);

        // Deduplicate by image_path (keep highest confidence for each image)
        let mut seen_paths: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        final_results.retain(|r| {
            let entry = seen_paths.entry(r.image_path.clone()).or_insert(0);
            if *entry == 0 {
                *entry = 1;
                true
            } else {
                false
            }
        });

        final_results.truncate(top_k);

        eprintln!("[ObjectSearch] Final {} results:", final_results.len());
        for (i, r) in final_results.iter().enumerate() {
            eprintln!("  [{}] image_id={}, confidence={:.4}, embedding={:.4}, inlier={:.4}, path={}",
                i, r.image_id, r.confidence, r.embedding_score, r.inlier_ratio, r.image_path);
        }

        Ok(final_results)
    }
}

/// Region of Interest
#[derive(Debug, Clone)]
pub struct Roi {
    pub bbox: [f32; 4],  // [x, y, w, h] in pixels
    pub region_type: &'static str,
    pub color_hist: Vec<f32>,
}

/// SuperPoint features
#[derive(Debug, Clone)]
pub struct SuperPointFeatures {
    pub width: u32,
    pub height: u32,
    pub keypoints: Vec<f32>,      // [x1, y1, x2, y2, ...]
    pub descriptors: Vec<f32>,    // [d1_1, d1_2, ..., d256_1, d256_2, ...]
}

/// LightGlue match result with additional metrics
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub num_inliers: usize,
    pub match_count: usize,
    pub inlier_ratio: f32,
    pub bbox_overlap: f32,
}