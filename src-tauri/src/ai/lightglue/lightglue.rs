//! LightGlue ONNX model wrapper for feature matching

use std::sync::Mutex;
use tracing::info;
use ort::session::Session;
use ort::value::Tensor;

use crate::core::features::normalize_keypoints;

/// LightGlue match result
#[derive(Debug, Clone)]
pub struct Match {
    /// Keypoint index in image 0
    pub kpt0_idx: usize,
    /// Keypoint index in image 1
    pub kpt1_idx: usize,
    /// Match score
    pub score: f32,
}

/// LightGlue matching result
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub matches: Vec<Match>,
    pub num_inliers: usize,
}

impl Default for MatchResult {
    fn default() -> Self {
        Self {
            matches: Vec::new(),
            num_inliers: 0,
        }
    }
}

/// LightGlue feature matcher
pub struct LightGlueMatcher {
    session: Mutex<Option<Session>>,
}

// Safety: LightGlueMatcher wraps Session in Mutex, which ensures thread-safe access
unsafe impl Send for LightGlueMatcher {}
unsafe impl Sync for LightGlueMatcher {}

impl LightGlueMatcher {
    pub fn new(model_path: &str) -> Result<Self, String> {
        info!("Loading LightGlue from: {}", model_path);
        let session = Session::builder()
            .expect("Failed to create session")
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load LightGlue: {}", e))?;

        info!("LightGlue loaded successfully");

        Ok(Self {
            session: Mutex::new(Some(session)),
        })
    }

    /// Match two sets of SuperPoint features
    ///
    /// # Arguments
    /// * `kpts0` - Keypoints for image 0, pixel coordinates [N, 2]
    /// * `desc0` - Descriptors for image 0 [N, 256]
    /// * `kpts1` - Keypoints for image 1, pixel coordinates [M, 2]
    /// * `desc1` - Descriptors for image 1 [M, 256]
    /// * `img0_size` - Image 0 size (width, height)
    /// * `img1_size` - Image 1 size (width, height)
    ///
    /// # Returns
    /// MatchResult with matches and scores
    pub fn match_features(
        &self,
        kpts0: &[f32],
        desc0: &[f32],
        kpts1: &[f32],
        desc1: &[f32],
        img0_size: (u32, u32),
        img1_size: (u32, u32),
    ) -> Result<MatchResult, String> {
        let mut guard = self.session.lock().unwrap();
        let session = guard.as_mut().ok_or("No session")?;

        let n0 = kpts0.len() / 2;
        let n1 = kpts1.len() / 2;

        if n0 == 0 || n1 == 0 {
            return Ok(MatchResult::default());
        }

        // Normalize keypoints to [-1, 1]
        let kpts0_norm = normalize_keypoints(kpts0, img0_size.0, img0_size.1);
        let kpts1_norm = normalize_keypoints(kpts1, img1_size.0, img1_size.1);

        // Stack into [2, N, 2] format
        let mut keypoints_data = Vec::with_capacity(2 * 1024 * 2);
        keypoints_data.extend_from_slice(&kpts0_norm);
        // Pad to 1024 if needed
        keypoints_data.resize(2 * 1024 * 2, 0.0_f32);
        // This is wrong, we need proper padding

        // Create properly padded inputs
        let kpts_stacked = self.stack_keypoints(&kpts0_norm, n0, &kpts1_norm, n1);
        let desc_stacked = self.stack_descriptors(desc0, n0, desc1, n1);

        // Create input tensors
        // LightGlue expects [2, max_keypoints, 2] keypoints and [2, max_keypoints, 256] descriptors
        const MAX_KPTS: usize = 1024;

        let kpts_shape = [2_i64, MAX_KPTS as i64, 2];
        let desc_shape = [2_i64, MAX_KPTS as i64, DESCRIPTOR_DIM as i64];

        let kpts_tensor = Tensor::from_array((kpts_shape, kpts_stacked))
            .map_err(|e| format!("Keypoints tensor error: {}", e))?;
        let desc_tensor = Tensor::from_array((desc_shape, desc_stacked))
            .map_err(|e| format!("Descriptors tensor error: {}", e))?;

        let outputs = session.run(ort::inputs![
            "keypoints" => kpts_tensor,
            "descriptors" => desc_tensor
        ]).map_err(|e| format!("Inference error: {}", e))?;

        // Parse output: matches [M, 3], mscores [M]
        let matches_flat = outputs[0].try_extract_tensor::<i64>()
            .map_err(|e| format!("Failed to extract matches: {}", e))?.1;
        let mscores_flat = outputs[1].try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract scores: {}", e))?.1;

        let num_matches = matches_flat.len() / 3;

        let mut matches = Vec::with_capacity(num_matches);
        for i in 0..num_matches {
            matches.push(Match {
                kpt0_idx: matches_flat[i * 3 + 1] as usize,
                kpt1_idx: matches_flat[i * 3 + 2] as usize,
                score: mscores_flat[i],
            });
        }

        // Filter by score threshold
        let threshold = 0.1;
        matches.retain(|m| m.score >= threshold);

        Ok(MatchResult {
            num_inliers: matches.len(),
            matches,
        })
    }

    /// Quick match with same image size
    pub fn match_features_same_size(
        &self,
        kpts0: &[f32],
        desc0: &[f32],
        kpts1: &[f32],
        desc1: &[f32],
        size: u32,
    ) -> Result<MatchResult, String> {
        self.match_features(kpts0, desc0, kpts1, desc1, (size, size), (size, size))
    }

    fn stack_keypoints(&self, kpts0: &[f32], n0: usize, kpts1: &[f32], n1: usize) -> Vec<f32> {
        const MAX_KPTS: usize = 1024;

        let mut result = vec![0.0_f32; 2 * MAX_KPTS * 2];

        // Fill image 0 keypoints
        for i in 0..n0.min(MAX_KPTS) {
            result[i * 2] = kpts0[i * 2];
            result[i * 2 + 1] = kpts0[i * 2 + 1];
        }

        // Fill image 1 keypoints
        for i in 0..n1.min(MAX_KPTS) {
            result[MAX_KPTS * 2 + i * 2] = kpts1[i * 2];
            result[MAX_KPTS * 2 + i * 2 + 1] = kpts1[i * 2 + 1];
        }

        result
    }

    fn stack_descriptors(&self, desc0: &[f32], n0: usize, desc1: &[f32], n1: usize) -> Vec<f32> {
        const MAX_KPTS: usize = 1024;
        const DESC_DIM: usize = 256;

        let mut result = vec![0.0_f32; 2 * MAX_KPTS * DESC_DIM];

        // Fill image 0 descriptors
        for i in 0..n0.min(MAX_KPTS) {
            for d in 0..DESC_DIM {
                result[i * DESC_DIM + d] = desc0[i * DESC_DIM + d];
            }
        }

        // Fill image 1 descriptors
        for i in 0..n1.min(MAX_KPTS) {
            for d in 0..DESC_DIM {
                result[MAX_KPTS * DESC_DIM + i * DESC_DIM + d] = desc1[i * DESC_DIM + d];
            }
        }

        result
    }
}

const DESCRIPTOR_DIM: usize = 256;

impl Drop for LightGlueMatcher {
    fn drop(&mut self) {
        info!("LightGlue session dropped");
    }
}
