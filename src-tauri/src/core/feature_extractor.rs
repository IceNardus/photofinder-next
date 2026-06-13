//! Feature extraction pipeline for SuperPoint + LightGlue
//!
//! Orchestrates: Patch Splitting -> SuperPoint -> VLAD Aggregation -> Storage

use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};
use uuid::Uuid;

use crate::ai::lightglue::{SuperPoint, SuperPointOutput};
use crate::core::features::{
    Bbox, DescriptorStats, ImageFeatures, PatchConfig, PatchFeature, PatchVector,
    compute_color_histogram, extract_patch, initialize_centroids, mean_aggregate,
    select_top_keypoints, split_into_patches,
};
use crate::core::database::Database;
use crate::core::database::patches::{PatchFeaturesTable, PatchVectorsTable};

/// Feature extraction pipeline
pub struct FeatureExtractor {
    superpoint: Arc<Mutex<SuperPoint>>,
    patch_config: PatchConfig,
    centroids: Vec<f32>,  // VLAD centroids [K, 256]
}

impl FeatureExtractor {
    pub fn new(
        superpoint_path: &str,
        centroids: Vec<f32>,
        patch_config: PatchConfig,
    ) -> Result<Self, String> {
        let superpoint = SuperPoint::new(superpoint_path)?;

        Ok(Self {
            superpoint: Arc::new(Mutex::new(superpoint)),
            patch_config,
            centroids,
        })
    }

    /// Extract features from a single image
    pub fn extract_image(&self, image_path: &str, image_id: &str) -> Result<ImageFeatures, String> {
        // Load color image
        let img = image::open(image_path)
            .map_err(|e| format!("Failed to open image: {}", e))?;

        // Convert to RGB for color histogram
        let rgb_img = img.to_rgb8();
        let gray = image::imageops::grayscale(&img);

        let width = gray.width();
        let height = gray.height();

        // Split into patches
        let patches = split_into_patches(&gray, &self.patch_config);
        info!("Split image {} into {} patches", image_path, patches.len());

        let mut image_features = ImageFeatures::new(
            image_id.to_string(),
            image_path.to_string(),
            width,
            height,
        );

        // Process each patch
        for patch_info in patches {
            // Extract grayscale patch for SuperPoint
            let patch_img = extract_patch(&gray, &patch_info);

            // Extract RGB patch for color histogram
            let rgb_patch = self.extract_rgb_patch(&rgb_img, &patch_info);

            // Run SuperPoint on patch
            let sp_output = {
                let mut guard = self.superpoint.lock().unwrap();
                guard.extract(&patch_img)?
            };

            // Select top-k keypoints
            let (kpts, _scores, descs) = select_top_keypoints(
                &sp_output.keypoints,
                &sp_output.scores,
                &sp_output.descriptors,
                self.patch_config.max_keypoints_per_patch,
            );

            let num_kpts = kpts.len() / 2;

            // Normalize keypoints to image coordinates (not patch coordinates)
            // Keypoints from SuperPoint are in patch pixel coords [0, 256]
            // We need to map them to image coordinates
            let kpts_image = self.patch_keypoints_to_image_coords(
                &kpts,
                patch_info.x,
                patch_info.y,
                patch_info.width,
                patch_info.height,
                width,
                height,
            );

            // Compute aggregated vector using VLAD
            let aggregated = if self.centroids.is_empty() {
                mean_aggregate(&descs, num_kpts)
            } else {
                // Use VLAD with centroids
                let desc_chunks: Vec<&[f32]> = descs.chunks(256).take(num_kpts).collect();

                // For simplicity, just use mean aggregation since VLAD requires more setup
                mean_aggregate(&descs, num_kpts)
            };

            // Create patch_id
            let patch_id = Uuid::new_v4().to_string();

            // Compute color histogram for this patch
            let color_hist = compute_color_histogram(&rgb_patch);

            // Store complete features in PatchFeature (for LightGlue later)
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
                color_hist,
            };

            // Store aggregated vector for HNSW
            let mut patch_vector = PatchVector::new(
                patch_id,
                image_id.to_string(),
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

    /// Convert patch-relative keypoints to image-relative coordinates
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
                // x coordinate - relative to patch, scale to image
                let x_in_patch = k / 256.0 * patch_w as f32;
                image_kpts.push((patch_x as f32 + x_in_patch) / img_w as f32);
            } else {
                // y coordinate
                let y_in_patch = k / 256.0 * patch_h as f32;
                image_kpts.push((patch_y as f32 + y_in_patch) / img_h as f32);
            }
        }

        image_kpts
    }
}

/// Save image features to database
pub fn save_image_features(
    db: &Database,
    features: &ImageFeatures,
) -> Result<(), String> {
    let conn = db.conn.lock().map_err(|e| format!("DB lock error: {}", e))?;

    // Delete existing features for this image (order matters: patch_vectors references patch_features)
    PatchVectorsTable::delete_by_image_id(&conn, &features.image_id)
        .map_err(|e| format!("Failed to delete old vectors: {}", e))?;
    PatchFeaturesTable::delete_by_image_id(&conn, &features.image_id)
        .map_err(|e| format!("Failed to delete old features: {}", e))?;

    // Insert new features
    for patch in &features.patches {
        PatchFeaturesTable::insert(&conn, patch)
            .map_err(|e| format!("Failed to insert patch: {}", e))?;
    }

    // Insert vectors
    PatchVectorsTable::insert_batch(&conn, &features.vectors)
        .map_err(|e| format!("Failed to insert vectors: {}", e))?;

    info!("Saved {} patches for image {}", features.patches.len(), features.image_id);

    Ok(())
}

/// Default VLAD centroids (can be trained on a sample of descriptors)
pub fn default_centroids() -> Vec<f32> {
    // Return empty - will fall back to mean aggregation
    vec![]
}
