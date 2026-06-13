//! Face Alignment to 112x112 RGB
//! Enhanced with tighter crop, lower face masking, and histogram equalization

use anyhow::Result;
use image::{GenericImageView, Rgb, RgbImage};
use std::f32::consts::PI;
use std::path::Path;

use super::detector::FiveKeypoints;

/// Face alignment configuration for AB testing
#[derive(Debug, Clone)]
pub struct FaceAlignmentConfig {
    /// Alignment scale factor (1.0 = no zoom, 1.25 = zoom in 25%)
    pub align_scale: f32,
    /// Y offset to shift landmarks up (negative = shift up)
    pub align_y_offset: f32,
    /// Apply ellipse face mask to remove clothes/shoulder
    pub use_ellipse_mask: bool,
    /// Apply histogram equalization for lighting robustness
    pub use_histogram_eq: bool,
    /// Chin Y threshold in 112x112 output - pixels below this are masked
    pub chin_mask_threshold: f32,
}

impl Default for FaceAlignmentConfig {
    fn default() -> Self {
        Self {
            align_scale: 1.25,
            align_y_offset: -10.0,
            use_ellipse_mask: true,
            use_histogram_eq: true,
            chin_mask_threshold: 98.0,
        }
    }
}

/// Config for AB test: mask ON, hist_eq OFF (recommended baseline)
pub fn mask_only_config() -> FaceAlignmentConfig {
    FaceAlignmentConfig {
        align_scale: 1.25,
        align_y_offset: -10.0,
        use_ellipse_mask: true,
        use_histogram_eq: false,
        chin_mask_threshold: 98.0,
    }
}

/// Config for AB test: mask OFF, hist_eq ON
pub fn hist_eq_only_config() -> FaceAlignmentConfig {
    FaceAlignmentConfig {
        align_scale: 1.25,
        align_y_offset: -10.0,
        use_ellipse_mask: false,
        use_histogram_eq: true,
        chin_mask_threshold: 98.0,
    }
}

/// Config for AB test: both OFF (raw alignment baseline)
pub fn raw_config() -> FaceAlignmentConfig {
    FaceAlignmentConfig {
        align_scale: 1.25,
        align_y_offset: -10.0,
        use_ellipse_mask: false,
        use_histogram_eq: false,
        chin_mask_threshold: 98.0,
    }
}

/// Metrics for verifying aligned face quality
#[derive(Debug, Clone)]
pub struct AlignedFaceMetrics {
    pub width: u32,
    pub height: u32,
    pub face_pixel_count: u32,
    pub face_coverage: f32,       // What % of image is non-black
    pub avg_luminance: f32,
    pub luminance_range: f32,
    pub is_valid: bool,            // True if face_coverage > 0.3 && avg_lum > 20.0
}

impl std::fmt::Display for AlignedFaceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedFaceMetrics({:.0}x{:.0} coverage={:.1}% lum={:.0} range={:.0} valid={})",
            self.width, self.height, self.face_coverage * 100.0, self.avg_luminance, self.luminance_range, self.is_valid)
    }
}

const OUTPUT_SIZE: u32 = 112;

/// Reference facial landmarks for 112x112 alignment (computed from config)
fn compute_ref_landmarks(config: &FaceAlignmentConfig) -> [(f32, f32); 5] {
    let scale = config.align_scale;
    let y_off = config.align_y_offset;
    [
        (38.2946 * scale, 51.6968 * scale + y_off),   // left eye
        (58.2946 * scale, 51.6968 * scale + y_off),    // right eye
        (50.2946 * scale, 87.0256 * scale + y_off),    // nose
        (41.5493 * scale, 107.3550 * scale + y_off),   // left mouth
        (61.7299 * scale, 107.3550 * scale + y_off),  // right mouth
    ]
}

pub struct FaceAligner {
    config: FaceAlignmentConfig,
}

impl FaceAligner {
    pub fn new() -> Self {
        Self::with_config(FaceAlignmentConfig::default())
    }

    pub fn with_config(config: FaceAlignmentConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &FaceAlignmentConfig {
        &self.config
    }

    /// Align face to 112x112 RGB using 5 keypoints
    pub fn align(&self, image_path: &str, keypoints: &FiveKeypoints) -> Result<RgbImage> {
        let img = image::open(image_path)?;
        self.align_from_image(&img, keypoints)
    }

    pub fn align_from_image(&self, img: &image::DynamicImage, keypoints: &FiveKeypoints) -> Result<RgbImage> {
        let src_points: [(f32, f32); 5] = [
            keypoints.left_eye,
            keypoints.right_eye,
            keypoints.nose,
            keypoints.left_mouth,
            keypoints.right_mouth,
        ];

        // Calculate similarity transform: src_points -> REF_LANDMARKS
        // transform = [m00, m01, t0, m10, m11, t1] where dst = M*src + t
        let ref_landmarks = compute_ref_landmarks(&self.config);
        let transform = self.compute_similarity_transform(&src_points, &ref_landmarks);

        // Create output image
        let mut output = RgbImage::new(OUTPUT_SIZE, OUTPUT_SIZE);

        // Transform maps: src = M^{-1} * (dst - t)
        // We need to compute the inverse to go from output dst coordinates to src coordinates
        let scale = (transform[0] * transform[0] + transform[1] * transform[1]).sqrt();
        let cos_theta = transform[0] / scale;
        let sin_theta = -transform[1] / scale; // R^T = R(-theta)

        // Inverse transform: src = (1/scale) * R^T * (dst - t)
        let r00 = transform[0];
        let r01 = transform[1];
        let r10 = transform[3];
        let r11 = transform[4];
        let t0 = transform[2];
        let t1 = transform[5];

        let inv_scale = 1.0 / scale;
        let inv_t0 = -inv_scale * (r00 * t0 + r10 * t1);
        let inv_t1 = -inv_scale * (r01 * t0 + r11 * t1);

        // Inverse rotation (transpose of R)
        let inv_m00 = r00 * inv_scale;
        let inv_m01 = r10 * inv_scale;
        let inv_m10 = r01 * inv_scale;
        let inv_m11 = r11 * inv_scale;

        // Apply inverse transform to sample from source
        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
                // Use inverse transform: src_point = M^{-1} * (dst_point - t)
                let src_x = inv_m00 * x as f32 + inv_m01 * y as f32 + inv_t0;
                let src_y = inv_m10 * x as f32 + inv_m11 * y as f32 + inv_t1;

                if src_x >= 0.0 && src_x < img.width() as f32 &&
                   src_y >= 0.0 && src_y < img.height() as f32 {
                    let px = src_x.floor() as u32;
                    let py = src_y.floor() as u32;
                    let fx = src_x - src_x.floor();
                    let fy = src_y - src_y.floor();

                    // Bilinear interpolation
                    let p00 = img.get_pixel(px.min(img.width()-1), py.min(img.height()-1));
                    let p10 = img.get_pixel((px+1).min(img.width()-1), py.min(img.height()-1));
                    let p01 = img.get_pixel(px.min(img.width()-1), (py+1).min(img.height()-1));
                    let p11 = img.get_pixel((px+1).min(img.width()-1), (py+1).min(img.height()-1));

                    let r = (p00[0] as f32 * (1.-fx) * (1.-fy) +
                            p10[0] as f32 * fx * (1.-fy) +
                            p01[0] as f32 * (1.-fx) * fy +
                            p11[0] as f32 * fx * fy) as u8;
                    let g = (p00[1] as f32 * (1.-fx) * (1.-fy) +
                            p10[1] as f32 * fx * (1.-fy) +
                            p01[1] as f32 * (1.-fx) * fy +
                            p11[1] as f32 * fx * fy) as u8;
                    let b = (p00[2] as f32 * (1.-fx) * (1.-fy) +
                            p10[2] as f32 * fx * (1.-fy) +
                            p01[2] as f32 * (1.-fx) * fy +
                            p11[2] as f32 * fx * fy) as u8;

                    output.put_pixel(x, y, Rgb([r, g, b]));
                } else {
                    output.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }

        // Step 2: Apply ellipse face mask (keeps only face region, removes clothes/shoulder)
        if self.config.use_ellipse_mask {
            self.apply_ellipse_face_mask(&mut output);
        }

        // Step 3: Apply histogram equalization for lighting robustness
        if self.config.use_histogram_eq {
            self.apply_histogram_equalization(&mut output);
        }

        Ok(output)
    }

    /// Apply elliptical mask centered on face - more natural than rectangular cutoff
    /// Only keeps pixels inside an ellipse covering the face, blacking out clothes/shoulder
    fn apply_ellipse_face_mask(&self, img: &mut RgbImage) {
        let center_x = OUTPUT_SIZE as f32 / 2.0;
        let center_y = OUTPUT_SIZE as f32 / 2.0 - 3.0;
        let radius_x = OUTPUT_SIZE as f32 * 0.40;
        let radius_y = OUTPUT_SIZE as f32 * 0.48;
        let chin_threshold = self.config.chin_mask_threshold;

        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
                let dx = (x as f32 - center_x) / radius_x;
                let dy = (y as f32 - center_y) / radius_y;
                let dist = dx * dx + dy * dy;
                if dist > 1.0 {
                    img.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
    }

    /// Apply histogram equalization to reduce lighting differences
    /// This makes embeddings more robust to illumination changes
    fn apply_histogram_equalization(&self, img: &mut RgbImage) {
        // Convert to grayscale for histogram computation
        let mut gray = vec![0u32; (OUTPUT_SIZE * OUTPUT_SIZE) as usize];

        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
                let pixel = img.get_pixel(x, y);
                // Luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
                let lum = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u32;
                gray[(y * OUTPUT_SIZE + x) as usize] = lum;
            }
        }

        // Compute histogram
        let mut hist = [0u32; 256];
        for &v in &gray {
            hist[v as usize] += 1;
        }

        // Compute CDF
        let total = (OUTPUT_SIZE * OUTPUT_SIZE) as u32;
        let mut cdf = [0u32; 256];
        cdf[0] = hist[0];
        for i in 1..256 {
            cdf[i] = cdf[i-1] + hist[i];
        }

        // Find first non-zero CDF for normalization
        let mut cdf_min = 0;
        for i in 0..256 {
            if cdf[i] > 0 {
                cdf_min = cdf[i];
                break;
            }
        }

        // Create equalization lookup table
        let mut lut = [0u8; 256];
        let denom = total - cdf_min;
        if denom > 0 {
            for i in 0..256 {
                lut[i] = ((cdf[i] - cdf_min) as f32 / denom as f32 * 255.0).round() as u8;
            }
        }

        // Apply equalization to each channel
        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
                let pixel = img.get_pixel(x, y);
                // Use luminance-based equalization but apply same transform to all channels
                let lum = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as usize;
                let eq_lum = lut[lum.min(255)] as f32;

                // Scale each channel by the equalized luminance ratio
                let orig_lum_f = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
                if orig_lum_f > 0.0 {
                    let ratio = eq_lum / orig_lum_f;
                    let r = (pixel[0] as f32 * ratio).clamp(0.0, 255.0) as u8;
                    let g = (pixel[1] as f32 * ratio).clamp(0.0, 255.0) as u8;
                    let b = (pixel[2] as f32 * ratio).clamp(0.0, 255.0) as u8;
                    img.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
        }
    }

    /// Compute 2D similarity transform: dst = M * src + t
    fn compute_similarity_transform(&self, src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> [f32; 6] {
        // Using Umeyama algorithm for similarity transform
        let mut src_center = (0.0f32, 0.0f32);
        let mut dst_center = (0.0f32, 0.0f32);
        for i in 0..5 {
            src_center.0 += src[i].0;
            src_center.1 += src[i].1;
            dst_center.0 += dst[i].0;
            dst_center.1 += dst[i].1;
        }
        src_center.0 /= 5.0;
        src_center.1 /= 5.0;
        dst_center.0 /= 5.0;
        dst_center.1 /= 5.0;

        // Normalize src points
        let mut src_norm = [(0.0f32, 0.0f32); 5];
        let mut dst_norm = [(0.0f32, 0.0f32); 5];
        let mut src_scale = 0.0f32;
        let mut dst_scale = 0.0f32;

        for i in 0..5 {
            src_norm[i].0 = src[i].0 - src_center.0;
            src_norm[i].1 = src[i].1 - src_center.1;
            dst_norm[i].0 = dst[i].0 - dst_center.0;
            dst_norm[i].1 = dst[i].1 - dst_center.1;
            src_scale += src_norm[i].0 * src_norm[i].0 + src_norm[i].1 * src_norm[i].1;
            dst_scale += dst_norm[i].0 * dst_norm[i].0 + dst_norm[i].1 * dst_norm[i].1;
        }
        src_scale = (src_scale / 5.0).sqrt();
        dst_scale = (dst_scale / 5.0).sqrt();

        let scale = dst_scale / src_scale;
        for i in 0..5 {
            src_norm[i].0 *= scale;
            src_norm[i].1 *= scale;
        }

        // Compute rotation (similarity transform)
        // Solve for optimal rotation using SVD of cross-covariance
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for i in 0..5 {
            a += src_norm[i].0 * dst_norm[i].0 + src_norm[i].1 * dst_norm[i].1;
            b += src_norm[i].0 * dst_norm[i].1 - src_norm[i].1 * dst_norm[i].0;
        }
        let norm = (a * a + b * b).sqrt();
        let cos_theta = a / norm;
        let sin_theta = b / norm;

        // Build transformation matrix
        // M = scale * R, where R is rotation matrix
        let m00 = scale * cos_theta;
        let m01 = scale * sin_theta;
        let m10 = -scale * sin_theta;
        let m11 = scale * cos_theta;

        let t0 = dst_center.0 - (m00 * src_center.0 + m01 * src_center.1);
        let t1 = dst_center.1 - (m10 * src_center.0 + m11 * src_center.1);

        [m00, m01, t0, m10, m11, t1]
    }

    /// Save aligned face to file for debug inspection
    pub fn save_debug(&self, aligned: &RgbImage, output_path: &str) -> Result<()> {
        aligned.save(output_path)?;
        Ok(())
    }

    /// Verify aligned face quality - returns metrics for debugging
    pub fn verify_aligned_face(&self, aligned: &RgbImage) -> AlignedFaceMetrics {
        let (w, h) = (aligned.width(), aligned.height());

        // Count non-black pixels (face region)
        let mut face_pixel_count = 0u32;
        let mut total_luminance = 0.0f32;
        let mut min_lum = 255.0f32;
        let mut max_lum = 0.0f32;

        for y in 0..h {
            for x in 0..w {
                let pixel = aligned.get_pixel(x, y);
                let lum = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
                if lum > 10.0 {
                    face_pixel_count += 1;
                }
                total_luminance += lum;
                min_lum = min_lum.min(lum);
                max_lum = max_lum.max(lum);
            }
        }

        let total_pixels = w * h;
        let face_coverage = face_pixel_count as f32 / total_pixels as f32;
        let avg_lum = total_luminance / total_pixels as f32;
        let lum_range = max_lum - min_lum;

        AlignedFaceMetrics {
            width: w,
            height: h,
            face_pixel_count,
            face_coverage,
            avg_luminance: avg_lum,
            luminance_range: lum_range,
            is_valid: face_coverage > 0.3 && avg_lum > 20.0,
        }
    }

    /// Save aligned face to debug directory for inspection
    pub fn save_for_debug(&self, aligned: &RgbImage, prefix: &str, image_id: i64) -> Result<String> {
        let debug_dir = Path::new("./debug/aligned_faces");
        std::fs::create_dir_all(debug_dir).map_err(|e| anyhow::anyhow!("Failed to create debug dir: {}", e))?;

        let filename = format!("{}_id{}_{:.0}x{:.0}.png", prefix, image_id, aligned.width(), aligned.height());
        let path = debug_dir.join(&filename);

        aligned.save(&path).map_err(|e| anyhow::anyhow!("Failed to save debug face: {}", e))?;
        Ok(path.to_string_lossy().to_string())
    }
}

/// Scan different alignment scales and return quality metrics for each
pub fn scan_alignment_scales(
    image_path: &str,
    keypoints: &FiveKeypoints,
    scales: &[f32],
) -> Vec<(f32, AlignedFaceMetrics)> {
    let img = match image::open(image_path) {
        Ok(i) => i,
        Err(_) => return vec![],
    };

    scales
        .iter()
        .map(|&scale| {
            let config = FaceAlignmentConfig {
                align_scale: scale,
                align_y_offset: -10.0,
                use_ellipse_mask: true,
                use_histogram_eq: false,
                chin_mask_threshold: 98.0,
            };
            let aligner = FaceAligner::with_config(config);
            let aligned = match aligner.align_from_image(&img, keypoints) {
                Ok(a) => a,
                Err(_) => return (scale, AlignedFaceMetrics::invalid()),
            };
            let metrics = aligner.verify_aligned_face(&aligned);
            (scale, metrics)
        })
        .collect()
}

/// Find optimal scale by maximizing face coverage and luminance range
pub fn find_optimal_scale(
    image_path: &str,
    keypoints: &FiveKeypoints,
) -> Option<(f32, AlignedFaceMetrics)> {
    let candidate_scales = [1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4];
    let results = scan_alignment_scales(image_path, keypoints, &candidate_scales);

    // Score = face_coverage * 0.7 + (luminance_range / 255.0) * 0.3
    let scored: Vec<(f32, f32)> = results
        .iter()
        .filter(|(_, m)| m.is_valid)
        .map(|(scale, metrics)| {
            let score = metrics.face_coverage * 0.7 + (metrics.luminance_range / 255.0) * 0.3;
            (*scale, score)
        })
        .collect();

    if scored.is_empty() {
        return None;
    }

    scored
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(scale, _)| {
            let results = scan_alignment_scales(image_path, keypoints, &[scale]);
            (scale, results.into_iter().next().unwrap().1)
        })
}

impl AlignedFaceMetrics {
    pub fn invalid() -> Self {
        Self {
            width: 112,
            height: 112,
            face_pixel_count: 0,
            face_coverage: 0.0,
            avg_luminance: 0.0,
            luminance_range: 0.0,
            is_valid: false,
        }
    }
}

impl Default for FaceAligner {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug: Save aligned face to ./debug/ directory
pub fn save_debug_aligned(aligned: &RgbImage, filename: &str) -> Result<(), String> {
    let debug_dir = std::path::Path::new("./debug");
    std::fs::create_dir_all(debug_dir).map_err(|e| e.to_string())?;

    let path = debug_dir.join(filename);
    aligned.save(&path).map_err(|e| e.to_string())?;

    Ok(())
}