use serde::{Deserialize, Serialize};

pub const PATCH_SIZE: u32 = 512;
pub const STRIDE: u32 = 256;

/// Bounding box in normalized coordinates [0, 1]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bbox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Bbox {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn to_array(&self) -> [f32; 4] {
        [self.x, self.y, self.w, self.h]
    }

    /// Compute IoU overlap with another bbox (normalized [0,1])
    pub fn overlap(&self, other: &Bbox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);
        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }
        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.w * self.h + other.w * other.h - intersection;
        if union <= 0.0 {
            return 0.0;
        }
        intersection / union
    }
}

/// A patch extracted from an image
#[derive(Debug, Clone)]
pub struct Patch<'a> {
    pub index: u8,
    pub bbox: Bbox,
    pub data: &'a [f32], // grayscale pixel data [H * W]
    pub width: u32,
    pub height: u32,
}

/// Single image's extracted features
#[derive(Debug, Clone)]
pub struct ImageFeatures {
    pub image_id: String,
    pub image_path: String,
    pub width: u32,
    pub height: u32,
    pub patches: Vec<PatchFeature>,
    pub vectors: Vec<PatchVector>,
    /// Normalized crop region for query images (used in spatial matching)
    pub crop_bbox: Option<Bbox>,
}

impl ImageFeatures {
    pub fn new(image_id: String, image_path: String, width: u32, height: u32) -> Self {
        Self {
            image_id,
            image_path,
            width,
            height,
            patches: Vec::new(),
            vectors: Vec::new(),
            crop_bbox: None,
        }
    }

    pub fn add_patch(&mut self, patch: PatchFeature, vector: PatchVector) {
        self.patches.push(patch);
        self.vectors.push(vector);
    }
}

/// Patch feature with keypoints and descriptors (stored in SQLite)
#[derive(Debug, Clone)]
pub struct PatchFeature {
    pub patch_id: String,
    pub image_id: String,
    pub patch_index: u8,
    pub keypoints: Vec<f32>,      // [N, 2] - normalized coordinates
    pub descriptors: Vec<f32>,    // [N, 256] - SuperPoint descriptors
    pub num_keypoints: usize,
    pub image_width: u32,
    pub image_height: u32,
    pub bbox: Bbox,
    pub color_hist: Vec<f32>,     // 64-dim HSV color histogram (H:8, S:4, V:2)
}

impl Default for PatchFeature {
    fn default() -> Self {
        Self {
            patch_id: String::new(),
            image_id: String::new(),
            patch_index: 0,
            keypoints: Vec::new(),
            descriptors: Vec::new(),
            num_keypoints: 0,
            image_width: 0,
            image_height: 0,
            bbox: Bbox::new(0.0, 0.0, 0.0, 0.0),
            color_hist: vec![0.0; 64],
        }
    }
}

/// Aggregated patch vector for HNSW indexing
#[derive(Debug, Clone)]
pub struct PatchVector {
    pub patch_id: String,
    pub image_id: String,
    pub patch_index: u8,
    pub vector: Vec<f32>,  // 256-dim aggregated descriptor
}

impl PatchVector {
    pub fn new(patch_id: String, image_id: String, patch_index: u8, vector: Vec<f32>) -> Self {
        Self {
            patch_id,
            image_id,
            patch_index,
            vector,
        }
    }

    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut self.vector {
                *v /= norm;
            }
        }
    }
}

/// Statistics for a set of descriptors
#[derive(Debug, Clone)]
pub struct DescriptorStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub count: usize,
}

impl DescriptorStats {
    pub fn compute(descriptors: &[f32], num_descriptors: usize) -> Self {
        if num_descriptors == 0 {
            return Self {
                mean: vec![0.0; 256],
                std: vec![0.0; 256],
                count: 0,
            };
        }

        let dim = 256;
        let mut mean = vec![0.0f32; dim];
        let mut variance = vec![0.0f32; dim];

        // Compute mean
        for desc in descriptors.chunks(dim).take(num_descriptors) {
            for (i, &v) in desc.iter().enumerate() {
                mean[i] += v;
            }
        }
        let n = num_descriptors as f32;
        for m in &mut mean {
            *m /= n;
        }

        // Compute variance
        for desc in descriptors.chunks(dim).take(num_descriptors) {
            for (i, &v) in desc.iter().enumerate() {
                let d = v - mean[i];
                variance[i] += d * d;
            }
        }
        for v in &mut variance {
            *v = (*v / n).sqrt();
        }

        Self {
            mean,
            std: variance,
            count: num_descriptors,
        }
    }
}

/// Configuration for patch splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchConfig {
    /// Patch size (width and height, square patches)
    pub patch_size: u32,
    /// Stride between patches
    pub stride: u32,
    /// Maximum keypoints to retain per patch
    pub max_keypoints_per_patch: usize,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patch_size: PATCH_SIZE,
            stride: STRIDE,
            max_keypoints_per_patch: 256,
        }
    }
}

impl PatchConfig {
    pub fn new(patch_size: u32, stride: u32, max_keypoints_per_patch: usize) -> Self {
        Self {
            patch_size,
            stride,
            max_keypoints_per_patch,
        }
    }
}

/// Patch splitting result
#[derive(Debug, Clone)]
pub struct SplitPatch {
    pub index: u8,
    pub bbox: Bbox,  // Normalized [0, 1] coordinates
    pub x: u32,      // Absolute pixel coordinates
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Split an image into patches using sliding window
/// Patch Size = 512, Stride = 256 (50% overlap)
pub fn split_into_patches(
    image: &image::GrayImage,
    config: &PatchConfig,
) -> Vec<SplitPatch> {
    let img_width = image.width();
    let img_height = image.height();

    let patch_size = config.patch_size;
    let stride = config.stride;

    // Calculate number of patches in each dimension
    let cols = if img_width <= patch_size {
        1
    } else {
        1 + (img_width.saturating_sub(patch_size) + stride - 1) / stride
    };
    let rows = if img_height <= patch_size {
        1
    } else {
        1 + (img_height.saturating_sub(patch_size) + stride - 1) / stride
    };

    let mut patches = Vec::new();
    let mut index = 0u8;

    for row in 0..rows {
        for col in 0..cols {
            // Calculate patch position
            let x = col * stride;
            let y = row * stride;

            // Ensure patch doesn't exceed image bounds
            let width = patch_size.min(img_width.saturating_sub(x));
            let height = patch_size.min(img_height.saturating_sub(y));

            if width < 10 || height < 10 {
                continue;
            }

            // Convert to normalized bbox [0, 1]
            let bbox = Bbox::new(
                x as f32 / img_width as f32,
                y as f32 / img_height as f32,
                width as f32 / img_width as f32,
                height as f32 / img_height as f32,
            );

            patches.push(SplitPatch {
                index,
                bbox,
                x,
                y,
                width,
                height,
            });

            index += 1;
        }
    }

    patches
}

/// Extract a patch from an image
pub fn extract_patch<'a>(
    image: &'a image::GrayImage,
    patch: &SplitPatch,
) -> image::GrayImage {
    use image::imageops;

    if patch.x + patch.width > image.width() || patch.y + patch.height > image.height() {
        // Return a copy if bounds are invalid
        return image::GrayImage::new(1, 1);
    }

    imageops::crop_imm(
        image,
        patch.x,
        patch.y,
        patch.width,
        patch.height,
    ).to_image()
}

/// Early exit check before LightGlue matching
pub fn early_exit_check(
    query_kpts_len: usize,
    query_desc_mean: &[f32],
    candidate_kpts_len: usize,
    candidate_desc_mean: &[f32],
    max_kpt_difference: usize,
    min_desc_similarity: f32,
) -> bool {
    // Check keypoint count difference
    if (query_kpts_len as i64 - candidate_kpts_len as i64).abs() as usize > max_kpt_difference {
        return true;
    }

    // Check descriptor mean similarity
    if query_desc_mean.len() != candidate_desc_mean.len() {
        return false;
    }

    let dot: f32 = query_desc_mean
        .iter()
        .zip(candidate_desc_mean.iter())
        .map(|(a, b)| a * b)
        .sum();

    dot < min_desc_similarity
}

/// Compute HSV color histogram for a patch
/// H: 0-360 -> 8 bins (each bin = 45 degrees)
/// S: 0-1 -> 4 bins
/// V: 0-1 -> 2 bins
/// Total: 8 * 4 * 2 = 64 dimensions
pub fn compute_color_histogram(rgb_image: &image::RgbImage) -> Vec<f32> {
    let mut hist = vec![0.0f32; 64];

    for pixel in rgb_image.pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        // RGB to HSV conversion
        let max_c = r.max(g).max(b);
        let min_c = r.min(g).min(b);
        let delta = max_c - min_c;

        // Value
        let v = max_c;

        // Saturation
        let s = if max_c > 1e-8 { delta / max_c } else { 0.0 };

        // Hue
        let h = if delta < 1e-8 {
            0.0
        } else if max_c == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max_c == g {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };

        let h = if h < 0.0 { h + 360.0 } else { h };

        // Bin indices
        let h_bin = ((h / 360.0) * 8.0) as usize % 8;
        let s_bin = (s * 4.0) as usize % 4;
        let v_bin = if v > 0.5 { 1 } else { 0 };

        let index = h_bin * 8 + s_bin * 2 + v_bin;
        hist[index] += 1.0;
    }

    // Normalize
    let sum: f32 = hist.iter().sum();
    if sum > 0.0 {
        for v in &mut hist {
            *v /= sum;
        }
    }

    hist
}

/// Compute histogram intersection similarity between two color histograms
pub fn histogram_intersection(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.min(*y))
        .sum()
}