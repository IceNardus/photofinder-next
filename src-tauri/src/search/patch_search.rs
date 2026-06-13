//! Patch-based image search using SuperPoint + LightGlue
//!
//! Two-stage retrieval:
//! 1. Coarse: HNSW search on VLAD-aggregated patch vectors
//! 2. Fine: LightGlue matching with RANSAC verification

use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};
use tracing::{info, warn};
use crate::core::database::Database;
use crate::core::features::{
    mean_aggregate, vlad_aggregate, select_top_keypoints,
    ImageFeatures, PatchFeature, PatchVector, Bbox,
    compute_color_histogram, histogram_intersection,
};
use crate::core::index::hnsw::{HnswIndex, SearchResult};
use crate::ai::lightglue::{LightGlueMatcher, MatchResult};
use crate::ai::lightglue::superpoint::SuperPoint;

const VECTOR_DIM: usize = 256;
const MAX_KEYPOINTS: usize = 1024;

/// Hash a patch_id string to i64 for HNSW indexing
fn hash_patch_id(patch_id: &str) -> i64 {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(patch_id.as_bytes());
    let hash = hasher.finalize();
    // Take first 8 bytes as i64
    let bytes: [u8; 8] = hash.as_bytes()[..8].try_into().unwrap();
    i64::from_le_bytes(bytes)
}

/// Aggregation method for descriptors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationMethod {
    /// Simple mean aggregation (baseline)
    Mean,
    /// VLAD (Vector of Locally Aggregated Descriptors)
    Vlad,
}

impl Default for AggregationMethod {
    fn default() -> Self {
        Self::Mean
    }
}

/// Patch search configuration
#[derive(Debug, Clone)]
pub struct PatchSearchConfig {
    /// Number of top candidates to retrieve from HNSW
    pub hnsw_top_k: usize,
    /// Maximum keypoints per patch for LightGlue
    pub max_keypoints_per_patch: usize,
    /// Minimum LightGlue matches to consider a match
    pub min_lightglue_matches: usize,
    /// Early exit: max keypoint count difference
    pub max_kpt_difference: usize,
    /// Early exit: minimum descriptor mean similarity
    pub min_desc_similarity: f32,
    /// RANSAC inlier threshold (pixels)
    pub ransac_threshold: f32,
    /// Minimum bbox overlap threshold for spatial matching (0.0 to 1.0)
    pub min_bbox_overlap: f32,
    /// Data directory for HNSW index
    pub data_dir: Option<String>,
    /// Aggregation method: Mean or Vlad
    pub aggregation_method: AggregationMethod,
    /// Path to VLAD centroids file (binary [K, 256] f32)
    pub vlad_centroids_path: Option<String>,
    /// Number of VLAD clusters (k-means k)
    pub vlad_k: usize,
}

impl Default for PatchSearchConfig {
    fn default() -> Self {
        Self {
            hnsw_top_k: 50,
            max_keypoints_per_patch: 256,
            min_lightglue_matches: 15,
            max_kpt_difference: 200,
            min_desc_similarity: 0.7,
            ransac_threshold: 4.0,
            min_bbox_overlap: 0.3,
            data_dir: None,
            aggregation_method: AggregationMethod::Mean,
            vlad_centroids_path: None,
            vlad_k: 64,
        }
    }
}

/// Patch search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchSearchResult {
    pub image_id: String,
    pub image_path: String,
    pub total_matches: usize,
    pub inlier_count: usize,
    pub confidence: f32,
    /// Number of query patches that matched (for Object Retrieval mode)
    pub matched_query_patches: usize,
}

/// Patch search engine
pub struct PatchSearch {
    db: Database,
    hnsw: Arc<Mutex<HnswIndex>>,
    superpoint: Arc<Mutex<SuperPoint>>,
    lightglue: Arc<Mutex<LightGlueMatcher>>,
    config: PatchSearchConfig,
    /// Reverse mapping from HNSW id (hash) to (patch_id, image_id)
    id_to_patch: std::sync::Mutex<std::collections::HashMap<i64, (String, String)>>,
    /// Path to HNSW index file
    index_path: String,
    /// VLAD centroids [K, 256], loaded from file
    vlad_centroids: Option<Vec<f32>>,
}

impl PatchSearch {
    pub fn new(
        db: Database,
        superpoint_path: &str,
        lightglue_path: &str,
        config: PatchSearchConfig,
    ) -> Result<Self, String> {
        let superpoint = SuperPoint::new(superpoint_path)?;
        let lightglue = LightGlueMatcher::new(lightglue_path)?;

        // Determine index path
        let index_path = if let Some(ref data_dir) = config.data_dir {
            std::path::Path::new(data_dir)
                .join("patch_hnsw.idx")
                .to_string_lossy()
                .to_string()
        } else {
            "data/patch_hnsw.idx".to_string()
        };

        // Load HNSW index
        let hnsw = HnswIndex::load(&index_path, VECTOR_DIM)
            .unwrap_or_else(|_| HnswIndex::new(VECTOR_DIM));

        // Load VLAD centroids if VLAD aggregation is enabled
        let vlad_centroids = if config.aggregation_method == AggregationMethod::Vlad {
            if let Some(ref centroids_path) = config.vlad_centroids_path {
                match load_vlad_centroids(centroids_path) {
                    Ok(centroids) => {
                        info!("Loaded VLAD centroids: {} clusters", centroids.len() / 256);
                        Some(centroids)
                    }
                    Err(e) => {
                        warn!("Failed to load VLAD centroids: {}, falling back to mean", e);
                        None
                    }
                }
            } else {
                warn!("VLAD aggregation enabled but no centroids path configured");
                None
            }
        } else {
            None
        };

        Ok(Self {
            db,
            hnsw: Arc::new(Mutex::new(hnsw)),
            superpoint: Arc::new(Mutex::new(superpoint)),
            lightglue: Arc::new(Mutex::new(lightglue)),
            config,
            id_to_patch: std::sync::Mutex::new(std::collections::HashMap::new()),
            index_path,
            vlad_centroids,
        })
    }

    /// Rebuild HNSW index from database
    pub fn rebuild_index(&self) -> Result<(), String> {
        let conn = self.db.conn.lock().map_err(|e| format!("DB lock error: {}", e))?;

        let patch_vectors = crate::core::database::patches::PatchVectorsTable::get_all(&conn)
            .map_err(|e| format!("Failed to load vectors: {}", e))?;

        drop(conn);

        let mut id_to_patch = self.id_to_patch.lock().unwrap();
        id_to_patch.clear();

        let mut hnsw = self.hnsw.lock().unwrap();
        hnsw.clear();

        for pv in patch_vectors {
            let hnsw_id = hash_patch_id(&pv.patch_id);
            hnsw.add(pv.vector, hnsw_id);
            id_to_patch.insert(hnsw_id, (pv.patch_id.clone(), pv.image_id.clone()));
        }

        hnsw.save(&self.index_path)
            .map_err(|e| format!("Failed to save HNSW: {}", e))?;

        info!("Rebuilt HNSW index with {} vectors", hnsw.len());
        Ok(())
    }

    /// Search for similar images using query image
    pub fn search(&self, query_path: &str, top_k: usize) -> Result<Vec<PatchSearchResult>, String> {
        // Step 1: Extract features from query image
        let query_features = self.extract_query_features(query_path)?;
        eprintln!("[SEARCH] Extracted {} patches", query_features.patches.len());

        if query_features.patches.is_empty() {
            warn!("No patches extracted from query image");
            return Ok(vec![]);
        }

        // Step 2: Coarse search - HNSW on VLAD vectors
        let candidates = self.coarse_search(&query_features)?;
        eprintln!("[SEARCH] Coarse search returned {} candidates", candidates.len());

        if candidates.is_empty() {
            info!("No candidates found in coarse search");
            return Ok(vec![]);
        }

        for (i, (img_id, score)) in candidates.iter().take(5).enumerate() {
            eprintln!("[SEARCH] Candidate {}: image_id={}, score={:.2}", i+1, img_id, score);
        }

        // Step 3: Fine search - LightGlue matching with early exit
        let results = self.fine_search(&query_features, candidates, top_k)?;

        Ok(results)
    }

    /// Search with crop region - only search within the cropped area
    pub fn search_with_crop(&self, query_path: &str, crop_bbox: [f32; 4], top_k: usize) -> Result<Vec<PatchSearchResult>, String> {
        // Step 1: Load and crop the image
        let img = image::open(query_path)
            .map_err(|e| format!("Failed to open image: {}", e))?;

        let width = img.width();
        let height = img.height();

        // crop_bbox is [x1, y1, x2, y2] in normalized coordinates [0, 1]
        let x1 = (crop_bbox[0] * width as f32) as u32;
        let y1 = (crop_bbox[1] * height as f32) as u32;
        let x2 = (crop_bbox[2] * width as f32) as u32;
        let y2 = (crop_bbox[3] * height as f32) as u32;

        // Ensure valid crop region
        let x1 = x1.min(width.saturating_sub(1));
        let y1 = y1.min(height.saturating_sub(1));
        let x2 = x2.max(x1 + 1).min(width);
        let y2 = y2.max(y1 + 1).min(height);

        let cropped = img.crop_imm(x1, y1, x2 - x1, y2 - y1);

        // Save cropped image to temp file for processing
        let temp_path = std::env::temp_dir().join(format!("patch_search_crop_{}.jpg", uuid::Uuid::new_v4()));
        cropped.save(&temp_path)
            .map_err(|e| format!("Failed to save crop: {}", e))?;

        // Step 2: Extract features from cropped image
        let query_features = self.extract_query_features(&temp_path.to_string_lossy())?;

        // Clean up temp file
        let _ = std::fs::remove_file(&temp_path);

        if query_features.patches.is_empty() {
            warn!("No patches extracted from cropped query");
            return Ok(vec![]);
        }

        // Step 3: Coarse search - HNSW on VLAD vectors
        let candidates = self.coarse_search(&query_features)?;

        if candidates.is_empty() {
            info!("No candidates found in coarse search");
            return Ok(vec![]);
        }

        // Step 4: Fine search - LightGlue matching with early exit
        let results = self.fine_search(&query_features, candidates, top_k)?;

        Ok(results)
    }

    /// Extract SuperPoint features from query image
    pub fn extract_query_features(&self, image_path: &str) -> Result<ImageFeatures, String> {
        let img = image::open(image_path)
            .map_err(|e| format!("Failed to open image: {}", e))?;

        // Convert to RGB for color histogram
        let rgb_img = img.to_rgb8();
        let gray = image::imageops::grayscale(&img);
        let width = gray.width();
        let height = gray.height();

        let patch_config = crate::core::features::PatchConfig::default();
        let patches = crate::core::features::split_into_patches(&gray, &patch_config);

        let mut image_features = ImageFeatures::new(
            "query".to_string(),
            image_path.to_string(),
            width,
            height,
        );

        for patch_info in patches {
            let patch_img = crate::core::features::extract_patch(&gray, &patch_info);

            // Extract RGB patch for color histogram
            let rgb_patch = self.extract_rgb_patch(&rgb_img, &patch_info);

            let sp_output = {
                let mut guard = self.superpoint.lock().unwrap();
                guard.extract(&patch_img)?
            };

            let (kpts, _scores, descs) = select_top_keypoints(
                &sp_output.keypoints,
                &sp_output.scores,
                &sp_output.descriptors,
                self.config.max_keypoints_per_patch,
            );

            let num_kpts = kpts.len() / 2;

            // Convert to image coordinates
            let kpts_image = self.patch_keypoints_to_image_coords(
                &kpts,
                patch_info.x,
                patch_info.y,
                patch_info.width,
                patch_info.height,
                width,
                height,
            );

            // Aggregate descriptors using configured method
            let aggregated = self.aggregate_descriptors(&descs, num_kpts);

            // Compute color histogram
            let color_hist = compute_color_histogram(&rgb_patch);

            let patch_id = format!("query_patch_{}", patch_info.index);

            let patch_feature = PatchFeature {
                patch_id: patch_id.clone(),
                image_id: "query".to_string(),
                patch_index: patch_info.index,
                keypoints: kpts_image,
                descriptors: descs,
                num_keypoints: num_kpts,
                image_width: width,
                image_height: height,
                bbox: patch_info.bbox,
                color_hist,
            };

            let mut patch_vector = PatchVector::new(
                patch_id,
                "query".to_string(),
                patch_info.index,
                aggregated,
            );
            patch_vector.normalize();

            image_features.add_patch(patch_feature, patch_vector);
        }

        Ok(image_features)
    }

    /// Extract an RGB patch from an RGB image
    fn extract_rgb_patch(&self, rgb_img: &image::RgbImage, patch: &crate::core::features::SplitPatch) -> image::RgbImage {
        use image::imageops;

        if patch.x + patch.width > rgb_img.width() || patch.y + patch.height > rgb_img.height() {
            return image::RgbImage::new(1, 1);
        }

        imageops::crop_imm(
            rgb_img,
            patch.x,
            patch.y,
            patch.width,
            patch.height,
        ).to_image()
    }

    fn patch_keypoints_to_image_coords(
        &self,
        patch_kpts: &[f32],
        patch_x: u32,
        patch_y: u32,
        patch_w: u32,
        patch_h: u32,
        img_w: u32,
        img_h: u32,
    ) -> Vec<f32> {
        let mut image_kpts = Vec::with_capacity(patch_kpts.len());

        for (i, &k) in patch_kpts.iter().enumerate() {
            if i % 2 == 0 {
                let x_in_patch = k / 256.0 * patch_w as f32;
                image_kpts.push((patch_x as f32 + x_in_patch) / img_w as f32);
            } else {
                let y_in_patch = k / 256.0 * patch_h as f32;
                image_kpts.push((patch_y as f32 + y_in_patch) / img_h as f32);
            }
        }

        image_kpts
    }

    /// Coarse search using HNSW
    /// Returns Vec<(image_id, aggregate_score)> sorted by score descending
    /// Each query patch only votes for its BEST match per candidate image (not all top_k)
    /// Select patches by keypoint count (richest first) instead of uniform sampling
    fn coarse_search(&self, query_features: &ImageFeatures) -> Result<Vec<(String, f32)>, String> {
        // Select top patches by keypoint count (most informative first)
        const MAX_QUERY_PATCHES: usize = 50;
        let total_patches = query_features.vectors.len();

        // Create index with keypoint count
        let mut patch_scores: Vec<(usize, usize)> = query_features.patches.iter()
            .enumerate()
            .map(|(idx, p)| (idx, p.num_keypoints))
            .collect();

        // Sort by keypoint count descending
        patch_scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top patches
        let selected_indices: Vec<usize> = patch_scores.iter()
            .take(MAX_QUERY_PATCHES)
            .map(|(idx, _)| *idx)
            .collect();

        if selected_indices.is_empty() {
            eprintln!("[COARSE] No patches selected!");
            return Ok(vec![]);
        }

        let query_vectors: Vec<Vec<f32>> = selected_indices.iter()
            .map(|&idx| query_features.vectors[idx].vector.clone())
            .collect();

        eprintln!("[COARSE] Using {} query patches (from {} total)", query_vectors.len(), total_patches);

        // Get the id_to_patch mapping (HNSW_id -> (patch_id, image_id))
        let id_to_patch = self.id_to_patch.lock().unwrap();
        eprintln!("[COARSE] id_to_patch size: {}", id_to_patch.len());

        // Track best score per candidate image per query patch
        let mut candidate_images: std::collections::HashMap<String, (f32, usize)> =
            std::collections::HashMap::new();

        for (q_idx, query_vec) in query_vectors.iter().enumerate() {
            let results = {
                let hnsw = self.hnsw.lock().unwrap();
                hnsw.search(query_vec, self.config.hnsw_top_k)
            };
            eprintln!("[COARSE] Query patch {}: {} HNSW results", q_idx, results.len());

            // For each query patch, find the best score per candidate image
            let mut best_per_image: std::collections::HashMap<String, f32> =
                std::collections::HashMap::new();

            for result in results {
                if let Some((_, image_id)) = id_to_patch.get(&result.id) {
                    let entry = best_per_image.entry(image_id.clone()).or_insert(0.0);
                    *entry = entry.max(result.score);
                }
            }

            // Add best scores to aggregate
            for (image_id, best_score) in best_per_image {
                let entry = candidate_images.entry(image_id).or_insert((0.0, 0));
                entry.0 += best_score;
                entry.1 += 1;  // Count how many query patches matched this image
            }
        }

        drop(id_to_patch);

        eprintln!("[COARSE] candidate_images map has {} entries", candidate_images.len());

        // Use raw accumulated score (more matching patches = higher score, not penalized)
        let mut candidates: Vec<(String, f32)> = candidate_images
            .into_iter()
            .map(|(k, (score, _matched_patches))| {
                // Raw score - don't normalize, more patches matching should give higher score
                (k, score)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("[COARSE] Found {} candidate images after normalization", candidates.len());

        // Return top candidates as Vec<(image_id, score)>
        let top_candidates: Vec<(String, f32)> = candidates
            .into_iter()
            .take(self.config.hnsw_top_k)
            .collect();

        Ok(top_candidates)
    }

    /// Fine search using LightGlue matching
    /// Supports two modes:
    /// - Object Retrieval (query_patch_count <= 2): Small cropped images, minimal filtering
    /// - Scene Retrieval (query_patch_count > 2): Large images, spatial consistency enforced
    fn fine_search(
        &self,
        query_features: &ImageFeatures,
        candidate_images: Vec<(String, f32)>,
        top_k: usize,
    ) -> Result<Vec<PatchSearchResult>, String> {
        let conn = self.db.conn.lock().map_err(|e| format!("DB lock error: {}", e))?;

        // Determine search mode based on query patch count
        let is_object_mode = query_features.patches.len() <= 2;
        let min_matches = if is_object_mode { 3 } else { self.config.min_lightglue_matches };

        eprintln!("[FINE] {} mode ({} query patches, min_matches={})",
              if is_object_mode { "Object Retrieval" } else { "Scene Retrieval" },
              query_features.patches.len(), min_matches);
        eprintln!("[FINE] Candidate images to check: {}", candidate_images.len());

        // Track results per image
        #[derive(Default)]
        struct ImageMatchResult {
            total_inliers: usize,
            total_matches: usize,
            coarse_score: f32,
            query_centers: Vec<(f32, f32)>,
            matched_query_patches: usize,  // Count of UNIQUE query patches that matched this image
            matched_query_patch_ids: std::collections::HashSet<u8>,  // Track which query patches already matched
            color_similarity_sum: f32,  // Sum of color similarity scores for matched patches
            overlap_sum: f32,  // Sum of bbox overlap scores for matched patches
            // For Object Retrieval: best match per query patch
            best_match_per_query_patch: std::collections::HashMap<u8, (usize, usize)>, // patch_index -> (inliers, matches)
        }

        let mut image_results: std::collections::HashMap<String, ImageMatchResult> =
            std::collections::HashMap::new();

        // Process each candidate image
        for (image_id, coarse_score) in candidate_images {
            let candidate_patches = crate::core::database::patches::PatchFeaturesTable::get_by_image_id(
                &conn, &image_id,
            ).unwrap_or_default();

            if candidate_patches.is_empty() {
                continue;
            }

            // Try matching each query patch with each candidate patch
            for query_patch in &query_features.patches {
                let mut best_inliers = 0;
                let mut best_matches = 0;
                let mut best_color_sim = 0.0f32;
                let mut best_overlap = 0.0f32;

                for candidate in &candidate_patches {
                    if query_patch.num_keypoints < 4 || candidate.num_keypoints < 4 {
                        continue;
                    }

                    // SPATIAL FILTER: Check bbox overlap - skip if spatially inconsistent
                    let overlap = query_patch.bbox.overlap(&candidate.bbox);
                    if overlap < self.config.min_bbox_overlap {
                        continue;
                    }

                    let match_result = {
                        let guard = self.lightglue.lock().unwrap();
                        guard.match_features(
                            &query_patch.keypoints,
                            &query_patch.descriptors,
                            &candidate.keypoints,
                            &candidate.descriptors,
                            (query_patch.image_width, query_patch.image_height),
                            (candidate.image_width, candidate.image_height),
                        )?
                    };

                    // Compute color similarity for this candidate
                    let color_sim = histogram_intersection(&query_patch.color_hist, &candidate.color_hist);

                    // Track best match for this query patch (for Object Retrieval)
                    if match_result.num_inliers > best_inliers {
                        best_inliers = match_result.num_inliers;
                        best_matches = match_result.matches.len();
                        best_color_sim = color_sim;
                        best_overlap = overlap;
                    }
                }

                // For Object Retrieval: only count the BEST match per query patch
                // For Scene Retrieval: accumulate all matches above threshold
                if best_inliers >= min_matches {
                    let entry = image_results.entry(image_id.clone()).or_insert(
                        ImageMatchResult { coarse_score, ..Default::default() }
                    );

                    if is_object_mode {
                        // Object Retrieval: take best match, don't accumulate
                        if entry.matched_query_patch_ids.insert(query_patch.patch_index) {
                            entry.matched_query_patches += 1;
                            entry.total_inliers += best_inliers;
                            entry.total_matches += best_matches;
                            entry.color_similarity_sum += best_color_sim;
                            entry.overlap_sum += best_overlap;
                            entry.query_centers.push((
                                query_patch.bbox.x + query_patch.bbox.w * 0.5,
                                query_patch.bbox.y + query_patch.bbox.h * 0.5,
                            ));
                            entry.best_match_per_query_patch.insert(query_patch.patch_index, (best_inliers, best_matches));
                        }
                    } else {
                        // Scene Retrieval: accumulate all matches above threshold
                        if entry.matched_query_patch_ids.insert(query_patch.patch_index) {
                            entry.matched_query_patches += 1;
                        }
                        entry.total_inliers += best_inliers;
                        entry.total_matches += best_matches;
                        entry.color_similarity_sum += best_color_sim;
                        entry.overlap_sum += best_overlap;
                        entry.query_centers.push((
                            query_patch.bbox.x + query_patch.bbox.w * 0.5,
                            query_patch.bbox.y + query_patch.bbox.h * 0.5,
                        ));
                    }
                    }
            }
        }

        info!("[FINE] Images with matches: {}", image_results.len());

        drop(conn);

        // Convert to final results
        let mut results: Vec<PatchSearchResult> = image_results.into_iter()
            .filter(|(_, r)| r.total_inliers >= min_matches)
            .map(|(image_id, r)| {
                let inlier_ratio = if r.total_matches > 0 {
                    r.total_inliers as f32 / r.total_matches as f32
                } else {
                    0.0
                };

                // match_quality_factor: what fraction of query patches matched this image
                // For Object Retrieval (1-2 patches), this is either 0.5 or 1.0
                let match_quality = r.matched_query_patches as f32 / query_features.patches.len() as f32;

                // avg_overlap: average bbox overlap for matched patches
                let avg_overlap = if r.matched_query_patches > 0 {
                    r.overlap_sum / r.matched_query_patches as f32
                } else {
                    0.0
                };

                // Color similarity: average over matched patches
                let avg_color_sim = if r.matched_query_patches > 0 {
                    r.color_similarity_sum / r.matched_query_patches as f32
                } else {
                    0.0
                };

                // Color multiplier: 0.5 (no color match) to 1.0 (perfect color match)
                let color_mult = 0.5 + 0.5 * avg_color_sim;

                // Overlap multiplier: reward spatially consistent matches
                let overlap_mult = 0.5 + 0.5 * avg_overlap;

                let confidence = if is_object_mode {
                    // Object Retrieval: sqrt-based formula with match quality, color and overlap bonus
                    // Higher confidence when more query patches matched, colors align, and spatially consistent
                    inlier_ratio * (r.total_inliers as f32).sqrt() * (1.0 + match_quality * 0.5) * color_mult * overlap_mult
                } else {
                    // Scene Retrieval: full formula with spatial penalty and overlap bonus
                    let spatial_penalty = if r.query_centers.len() >= 3 {
                        let variance = Self::compute_spatial_variance(&r.query_centers);
                        (1.0 - variance).max(0.3)
                    } else {
                        1.0
                    };
                    let patch_bonus = (1.0 + r.query_centers.len() as f32).ln().max(1.0);
                    let base_score = r.total_inliers as f32 * inlier_ratio;
                    base_score * spatial_penalty * patch_bonus * color_mult * overlap_mult
                };

                PatchSearchResult {
                    image_id,
                    image_path: String::new(),
                    total_matches: r.total_matches,
                    inlier_count: r.total_inliers,
                    confidence,
                    matched_query_patches: r.matched_query_patches,
                }
            })
            .collect();

        // Sort by confidence descending
        // For Object Retrieval mode, use secondary sort by matched_query_patches
        // to prefer candidates that matched more query patches
        if is_object_mode {
            results.sort_by(|a, b| {
                match b.confidence.partial_cmp(&a.confidence).unwrap() {
                    std::cmp::Ordering::Equal => {
                        // Secondary: prefer more matched patches
                        b.matched_query_patches.cmp(&a.matched_query_patches)
                    }
                    other => other,
                }
            });
        } else {
            results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        }
        results.truncate(top_k);

        // Look up image paths
        for r in &mut results {
            if let Ok(Some(img)) = crate::core::database::images::ImagesTable::get_by_id(
                &self.db.conn, r.image_id.parse::<i64>().unwrap_or(0)
            ) {
                r.image_path = img.path;
            }
        }

        Ok(results)
    }

/// Index an image (offline pipeline)
    pub fn index_image(&self, image_path: &str, image_id: &str) -> Result<ImageFeatures, String> {
        let img = image::open(image_path)
            .map_err(|e| format!("Failed to open image: {}", e))?;

        let gray = image::imageops::grayscale(&img);
        let width = gray.width();
        let height = gray.height();

        let patch_config = crate::core::features::PatchConfig::default();
        let patches = crate::core::features::split_into_patches(&gray, &patch_config);

        let mut image_features = ImageFeatures::new(
            image_id.to_string(),
            image_path.to_string(),
            width,
            height,
        );

        for patch_info in patches {
            let patch_img = crate::core::features::extract_patch(&gray, &patch_info);

            let sp_output = {
                let mut guard = self.superpoint.lock().unwrap();
                guard.extract(&patch_img)?
            };

            let (kpts, scores, descs) = select_top_keypoints(
                &sp_output.keypoints,
                &sp_output.scores,
                &sp_output.descriptors,
                self.config.max_keypoints_per_patch,
            );

            let num_kpts = kpts.len() / 2;

            let kpts_image = self.patch_keypoints_to_image_coords(
                &kpts,
                patch_info.x,
                patch_info.y,
                patch_info.width,
                patch_info.height,
                width,
                height,
            );

            let aggregated = self.aggregate_descriptors(&descs, num_kpts);

            let patch_id = uuid::Uuid::new_v4().to_string();
            let patch_id_for_hnsw = patch_id.clone();

            let patch_feature = PatchFeature {
                patch_id: patch_id.clone(),
                image_id: image_id.to_string(),
                patch_index: patch_info.index,
                keypoints: kpts_image,
                descriptors: descs,
                num_keypoints: num_kpts,
                image_width: width,
                image_height: height,
                bbox: patch_info.bbox,
                color_hist: vec![0.0; 64],  // TODO: compute actual histogram
            };

            let mut patch_vector = PatchVector::new(
                patch_id,
                image_id.to_string(),
                patch_info.index,
                aggregated,
            );
            patch_vector.normalize();

            // Add to HNSW
            {
                let mut hnsw = self.hnsw.lock().unwrap();
                let mut id_to_patch = self.id_to_patch.lock().unwrap();
                let hnsw_id = hash_patch_id(&patch_id_for_hnsw);
                hnsw.add(patch_vector.vector.clone(), hnsw_id);
                id_to_patch.insert(hnsw_id, (patch_id_for_hnsw.clone(), image_id.to_string()));
            }

            image_features.add_patch(patch_feature, patch_vector);
        }

        // Save to database
        crate::core::feature_extractor::save_image_features(&self.db, &image_features)?;

        info!("Indexed image {} with {} patches", image_id, image_features.patches.len());

        Ok(image_features)
    }

    /// Save HNSW index to disk
    pub fn save_index(&self) -> Result<(), String> {
        let hnsw = self.hnsw.lock().unwrap();
        hnsw.save(&self.index_path)
            .map_err(|e| format!("Failed to save HNSW: {}", e))
    }

    /// Compute spatial variance of patch centers
    /// Low variance = patches are clustered together (good match)
    /// High variance = patches are scattered (likely false positive)
    fn compute_spatial_variance(centers: &[(f32, f32)]) -> f32 {
        if centers.is_empty() {
            return 0.0;
        }

        let n = centers.len() as f32;
        let mut mean_x = 0.0f32;
        let mut mean_y = 0.0f32;

        for &(x, y) in centers {
            mean_x += x;
            mean_y += y;
        }
        mean_x /= n;
        mean_y /= n;

        let mut variance = 0.0f32;
        for &(x, y) in centers {
            let dx = x - mean_x;
            let dy = y - mean_y;
            variance += dx * dx + dy * dy;
        }
        variance /= n;

        variance
    }

    /// Aggregate descriptors using configured method (Mean or VLAD)
    fn aggregate_descriptors(&self, descriptors: &[f32], num_descriptors: usize) -> Vec<f32> {
        match self.config.aggregation_method {
            AggregationMethod::Vlad => {
                if let Some(ref centroids) = self.vlad_centroids {
                    vlad_aggregate(
                        descriptors,
                        num_descriptors,
                        self.config.max_keypoints_per_patch,
                        centroids,
                    )
                } else {
                    // Fallback to mean if no centroids loaded
                    warn!("VLAD requested but no centroids available, using mean");
                    mean_aggregate(descriptors, num_descriptors)
                }
            }
            AggregationMethod::Mean => {
                mean_aggregate(descriptors, num_descriptors)
            }
        }
    }
}

/// Load VLAD centroids from binary file [K, 256] f32
fn load_vlad_centroids(path: &str) -> Result<Vec<f32>, String> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open centroids file {}: {}", path, e))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| format!("Failed to read centroids file: {}", e))?;

    if buffer.len() % 4 != 0 {
        return Err(format!(
            "Invalid centroids file size: {} bytes (not divisible by 4)",
            buffer.len()
        ));
    }

    let num_floats = buffer.len() / 4;
    let mut centroids = vec![0.0f32; num_floats];

    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.as_ptr() as *const f32,
            centroids.as_mut_ptr(),
            num_floats,
        );
    }

    let k = num_floats / 256;
    if k == 0 {
        return Err("Centroids file too small".to_string());
    }

    info!("Loaded {} VLAD centroids ({} clusters)", num_floats, k);
    Ok(centroids)
}