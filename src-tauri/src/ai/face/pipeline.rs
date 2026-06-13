//! FacePipeline MVP - Combines Detector + Aligner + ArcFace

use std::sync::Arc;
use std::path::Path;
use anyhow::Result;
use tracing::info;

use super::{FaceDetector, FaceAligner, ArcFace, FaceFeature, DetectedFace};
use crate::ai::debug::{DebugConfig, save_detection_overlay, save_aligned_face_debug};

const QUALITY_WEIGHT_DETECTOR: f32 = 0.7;
const QUALITY_WEIGHT_BLUR: f32 = 0.0;
const QUALITY_WEIGHT_POSE: f32 = 0.3;

const EMBEDDING_DIM: usize = 512;

pub struct FacePipeline {
    detector: FaceDetector,
    aligner: FaceAligner,
    arcface: ArcFace,
    debug_config: Option<DebugConfig>,
}

impl FacePipeline {
    pub fn new(
        detector: FaceDetector,
        aligner: FaceAligner,
        arcface: ArcFace,
    ) -> Self {
        Self {
            detector,
            aligner,
            arcface,
            debug_config: None,
        }
    }

    pub fn with_debug(detector: FaceDetector, aligner: FaceAligner, arcface: ArcFace, debug_dir: &Path) -> Self {
        Self {
            detector,
            aligner,
            arcface,
            debug_config: Some(DebugConfig::new(debug_dir)),
        }
    }

    /// Process a single image and return face features
    pub fn process_image(&self, image_path: &str) -> Result<Vec<FaceFeature>> {
        info!("Processing image: {}", image_path);

        // Step 1: Detect faces with SCRFD
        let detected_faces = self.detector.detect(image_path)
            .map_err(|e| anyhow::anyhow!("Detection failed: {}", e))?;

        info!("Detected {} faces", detected_faces.len());

        // Debug: save detection overlay
        if let Some(ref debug_config) = self.debug_config {
            save_detection_overlay(image_path, &detected_faces, debug_config);
        }

        // Get image dimensions for face_area_score calculation
        let img = image::open(image_path)?;
        let (img_width, img_height) = (img.width(), img.height());

        // Statistics counters for each filtering stage
        let mut stats_scrfd = 0;
        let mut stats_alignment_failed = 0;
        let mut stats_embedding_failed = 0;
        let mut stats_detector_low = 0;
        let mut stats_area_small = 0;
        let mut stats_eye_close = 0;
        let mut stats_pose_bad = 0;

        let mut features = Vec::new();

        for (i, face) in detected_faces.iter().enumerate() {
            stats_scrfd += 1;

            // Step 2: Align face to 112x112
            let aligned = match self.aligner.align_from_image(&img, &face.keypoints) {
                Ok(a) => a,
                Err(e) => {
                    stats_alignment_failed += 1;
                    continue;
                }
            };

            // Step 3: Extract ArcFace embedding
            let mut embedding = match self.arcface.extract(&aligned) {
                Ok(e) => e,
                Err(e) => {
                    stats_embedding_failed += 1;
                    continue;
                }
            };

            // Step 3.5: L2 normalize the embedding for consistent similarity computation
            Self::l2_normalize(&mut embedding);

            // Step 4: Calculate quality score
            let (detector_score, blur_score, pose_score, face_area_score, quality) =
                self.calculate_quality(&face, &aligned, img_width, img_height);

            // Filter 1: detector_score < 0.3 (放宽，与 SCRFD 修改一致)
            if detector_score < 0.3 {
                stats_detector_low += 1;
                continue;
            }
            // Filter 2: face_area_score < 0.15 (放宽)
            if face_area_score < 0.15 {
                stats_area_small += 1;
                continue;
            }
            // Filter 3: eye_distance < 12.0 pixels (放宽)
            let eye_dist = self.compute_eye_distance(&face);
            if eye_dist < 12.0 {
                stats_eye_close += 1;
                continue;
            }
            // Filter 4: pose_score < 0.2 (放宽)
            if pose_score < 0.2 {
                stats_pose_bad += 1;
                continue;
            }

            features.push(FaceFeature::new(
                face.bbox,
                face.score,
                blur_score,
                pose_score,
                face_area_score,
                quality,
                embedding,
            ));
        }

        // Print filtering statistics
        info!("[FACE_STATS] {} | align_fail={} | embed_fail={} | det_low={} | area_small={} | eye_close={} | pose_bad={} | final={}",
              stats_scrfd, stats_alignment_failed, stats_embedding_failed,
              stats_detector_low, stats_area_small, stats_eye_close, stats_pose_bad, features.len());
        info!("Extracted {} face features after filtering", features.len());
        Ok(features)
    }

    fn calculate_quality(&self, face: &DetectedFace, aligned: &image::RgbImage, img_width: u32, img_height: u32) -> (f32, f32, f32, f32, f32) {
        // detector_score: SCRFD confidence (0-1)
        let detector_score = face.score;

        // blur_score: computed from Laplacian variance on aligned face
        let blur_score = self.compute_blur_score_from_aligned(aligned);

        // pose_score: estimated from 5 keypoints symmetry
        let pose_score = self.compute_pose_score(face);

        // min_face_size: absolute face dimension (more stable than area ratio)
        let face_w = face.bbox[2] - face.bbox[0];
        let face_h = face.bbox[3] - face.bbox[1];
        let min_face_size = face_w.min(face_h);

        // eye_distance: distance between eyes (stable across image sizes)
        let eye_dist = self.compute_eye_distance(face);

        // face_area_score: normalized, but using absolute min dimension
        // 60px = poor, 100px = medium, 150px+ = good
        let face_area_score = (min_face_size / 150.0).clamp(0.0, 1.0);

        // combined quality score
        let quality = detector_score * QUALITY_WEIGHT_DETECTOR +
            blur_score * QUALITY_WEIGHT_BLUR +
            pose_score * QUALITY_WEIGHT_POSE;

        (detector_score, blur_score, pose_score, face_area_score, quality)
    }

    fn compute_blur_score_from_aligned(&self, aligned: &image::RgbImage) -> f32 {
        let gray = image::DynamicImage::ImageRgb8(aligned.clone()).to_luma8();
        let variance = self.laplacian_variance(&gray);
        // Normalize: variance > 100 = sharp, variance < 30 = blurry
        (variance / 100.0).clamp(0.0, 1.0)
    }

    fn laplacian_variance(&self, gray: &image::GrayImage) -> f32 {
        let mut sum = 0.0_f32;
        let mut sum_sq = 0.0_f32;
        let mut count = 0.0_f32;

        for y in 1..gray.height().saturating_sub(1) {
            for x in 1..gray.width().saturating_sub(1) {
                let center = gray.get_pixel(x, y)[0] as f32;
                let laplacian =
                    4.0 * center
                    - gray.get_pixel(x.saturating_sub(1), y)[0] as f32
                    - gray.get_pixel((x + 1).min(gray.width() - 1), y)[0] as f32
                    - gray.get_pixel(x, y.saturating_sub(1))[0] as f32
                    - gray.get_pixel(x, (y + 1).min(gray.height() - 1))[0] as f32;
                sum += laplacian;
                sum_sq += laplacian * laplacian;
                count += 1.0;
            }
        }

        if count < 1.0 { return 0.0; }
        let mean = sum / count;
        (sum_sq / count) - (mean * mean)
    }

    fn compute_pose_score(&self, face: &DetectedFace) -> f32 {
        // Estimate pose from 5 keypoints symmetry
        let kps = &face.keypoints;
        let left_eye = kps.left_eye;
        let right_eye = kps.right_eye;
        let left_mouth = kps.left_mouth;
        let right_mouth = kps.right_mouth;

        // Eye distance ratio
        let eye_dist_x = (right_eye.0 - left_eye.0).abs();
        let eye_dist_y = (right_eye.1 - left_eye.1).abs();
        let eye_tilt = eye_dist_y / eye_dist_x.max(1.0);

        // Mouth distance ratio
        let mouth_dist_x = (right_mouth.0 - left_mouth.0).abs();
        let mouth_dist_y = (right_mouth.1 - left_mouth.1).abs();
        let mouth_tilt = mouth_dist_y / mouth_dist_x.max(1.0);

        // Symmetry score based on tilt angles
        let tilt_score = 1.0 - (eye_tilt + mouth_tilt).min(1.0);

        // Eye level: left and right eye should be roughly at same height
        let eye_level_diff = (left_eye.1 - right_eye.1).abs();
        let eye_level_score = 1.0 - (eye_level_diff / 50.0).min(1.0);

        (tilt_score + eye_level_score) / 2.0
    }

    fn compute_eye_distance(&self, face: &DetectedFace) -> f32 {
        let kps = &face.keypoints;
        let dx = kps.right_eye.0 - kps.left_eye.0;
        let dy = kps.right_eye.1 - kps.left_eye.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    /// L2 normalize a vector in-place
    fn l2_normalize(v: &mut Vec<f32>) {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
