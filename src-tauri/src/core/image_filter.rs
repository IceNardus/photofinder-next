//! Image filter to skip non-photo images before face pipeline
//! Filters out: icons, screenshots, illustrations, thumbnails, etc.

use std::path::Path;
use image::GenericImageView;
use tracing::info;

const MIN_WIDTH_HEIGHT: u32 = 256;
const MIN_BPP: f32 = 0.05;  // bytes per pixel, below this = likely icon/vector
const MAX_ASPECT_RATIO: f32 = 5.0;  // extreme wide/tall = likely screenshot

/// Returns true if the image should be skipped for face pipeline
pub fn should_skip_for_face_pipeline(path: &Path, file_size: u64) -> (bool, String) {
    // Try to open and analyze image content
    if let Ok(img) = image::open(path) {
        let (w, h) = img.dimensions();

        // 1. Size filtering - skip very small images (icons, UI elements)
        if w < MIN_WIDTH_HEIGHT || h < MIN_WIDTH_HEIGHT {
            return (true, format!("too_small_{}x{}", w, h));
        }

        // 2. Aspect ratio filtering - extreme ratios are likely screenshots
        let ratio = w as f32 / h as f32;
        if ratio < 1.0 / MAX_ASPECT_RATIO || ratio > MAX_ASPECT_RATIO {
            return (true, format!("bad_aspect_ratio_{:.2}", ratio));
        }

        // 3. Bytes per pixel filtering
        let pixels = (w * h) as f32;
        let bpp = file_size as f32 / pixels;
        if bpp < MIN_BPP {
            return (true, format!("low_bpp_{:.4}", bpp));
        }

        // 4. Color entropy filtering - low entropy = simple graphics
        let entropy = compute_color_entropy(&img);
        if entropy < 3.5 {  // Simple graphics typically have entropy < 3.5
            return (true, format!("low_entropy_{:.2}", entropy));
        }
    } else {
        // Can't open image - skip it
        return (true, "cannot_open_image".to_string());
    }

    (false, "ok".to_string())
}

/// Compute approximate color entropy of image (sampled for performance)
fn compute_color_entropy(img: &image::DynamicImage) -> f32 {
    // Resize for performance
    let sampled = img.resize(80, 80, image::imageops::FilterType::Nearest);
    let rgba = sampled.to_rgba8();

    // Count color frequencies
    let mut color_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    for pixel in rgba.pixels() {
        // Pack RGB into u24
        let rgb = ((pixel[0] as u32) << 16) | ((pixel[1] as u32) << 8) | (pixel[2] as u32);
        *color_counts.entry(rgb).or_insert(0) += 1;
    }

    let total = rgba.pixels().count() as f32;

    // Compute Shannon entropy
    let mut entropy = 0.0_f32;
    for &count in color_counts.values() {
        let p = count as f32 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}
