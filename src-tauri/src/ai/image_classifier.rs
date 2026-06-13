//! Image Type Classifier - Filters out screenshots, documents, posters, anime, etc.

use std::path::Path;
use image::GenericImageView;
use tracing::info;

/// Classified image type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageType {
    Photo,       // Real photo - proceed with face detection
    Screenshot,  // Screen capture
    Document,    // Documents, receipts, etc.
    Poster,      // Posters, cards, etc.
    Meme,        // Internet memes
    Anime,       // Anime/CG illustrations
    Unknown,     // Could not classify
}

impl ImageType {
    pub fn should_process(&self) -> bool {
        matches!(self, ImageType::Photo)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ImageType::Photo => "photo",
            ImageType::Screenshot => "screenshot",
            ImageType::Document => "document",
            ImageType::Poster => "poster",
            ImageType::Meme => "meme",
            ImageType::Anime => "anime",
            ImageType::Unknown => "unknown",
        }
    }
}

/// Image type classifier
pub struct ImageTypeClassifier {
    min_width_height: u32,
    max_aspect_ratio: f32,
    min_entropy: f32,
    max_saturation: f32,
}

impl ImageTypeClassifier {
    pub fn new() -> Self {
        Self {
            min_width_height: 256,
            max_aspect_ratio: 2.5,       // More extreme = screenshot (tightened)
            min_entropy: 4.0,             // Photos typically > 4.0, anime 3.0-3.5
            max_saturation: 0.65,         // Photos typically < 0.65, anime/poster > 0.7
        }
    }

    /// Classify an image file
    pub fn classify_path(&self, path: &Path) -> (ImageType, String) {
        // Try to open and analyze image content
        if let Ok(img) = image::open(path) {
            self.classify_image(&img)
        } else {
            (ImageType::Unknown, "cannot_open_image".to_string())
        }
    }

    /// Classify an already-loaded image
    pub fn classify_image(&self, img: &image::DynamicImage) -> (ImageType, String) {
        let (w, h) = img.dimensions();

        // 1. Size filtering - too small = icon/UI
        if w < self.min_width_height || h < self.min_width_height {
            return (ImageType::Unknown, format!("too_small_{}x{}", w, h));
        }

        // 2. Aspect ratio - extreme = screenshot
        let ratio = w as f32 / h as f32;
        if ratio < 1.0 / self.max_aspect_ratio || ratio > self.max_aspect_ratio {
            return (ImageType::Screenshot, format!("extreme_aspect_ratio_{:.2}", ratio));
        }

        // 3. Resize for analysis
        let sample = img.resize(120, 120, image::imageops::FilterType::Nearest);
        let rgba = sample.to_rgba8();

        // 4. Compute features
        let entropy = self.compute_entropy(&rgba);
        let avg_saturation = self.compute_avg_saturation(&rgba);
        let color_count = self.compute_color_count(&rgba);

        // 5. Classification logic
        // Low entropy alone is enough to suspect non-photo (anime/illustration)
        if entropy < 4.0 {
            return (ImageType::Anime, format!("low_entropy{:.2}_sat{:.2}", entropy, avg_saturation));
        }

        // High saturation alone is suspicious
        if avg_saturation > self.max_saturation {
            return (ImageType::Poster, format!("high_saturation{:.2}", avg_saturation));
        }

        // Low color count = poster/meme
        if color_count < 100 {
            return (ImageType::Meme, format!("low_colors{}", color_count));
        }

        // Low brightness variation = screenshot UI
        let brightness_var = self.compute_brightness_variance(&rgba);
        if brightness_var < 0.01 && (ratio > 1.5 || ratio < 0.67) {
            return (ImageType::Screenshot, format!("low_brightness_var{:.4}", brightness_var));
        }

        // All checks passed = likely photo
        (ImageType::Photo, format!("entropy={:.2}, sat={:.2}, colors={}", entropy, avg_saturation, color_count))
    }

    fn compute_entropy(&self, rgba: &image::RgbaImage) -> f32 {
        let mut color_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

        for pixel in rgba.pixels() {
            // Pack RGB into u24, ignore alpha
            let rgb = ((pixel[0] as u32) << 16) | ((pixel[1] as u32) << 8) | (pixel[2] as u32);
            *color_counts.entry(rgb).or_insert(0) += 1;
        }

        let total = rgba.pixels().count() as f32;
        let mut entropy = 0.0_f32;

        for &count in color_counts.values() {
            let p = count as f32 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    fn compute_avg_saturation(&self, rgba: &image::RgbaImage) -> f32 {
        let mut total_sat = 0.0f32;
        let mut count = 0u32;

        for pixel in rgba.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let l = (max + min) / 2.0;

            if l > 0.0 && l < 1.0 {
                let d = max - min;
                let s = if l < 0.5 {
                    d / (max + min)
                } else {
                    d / (2.0 - max - min)
                };
                total_sat += s;
                count += 1;
            }
        }

        if count > 0 {
            total_sat / count as f32
        } else {
            0.0
        }
    }

    fn compute_color_count(&self, rgba: &image::RgbaImage) -> usize {
        let mut colors: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for pixel in rgba.pixels() {
            // Quantize to reduce noise
            let r = (pixel[0] / 16) as u32;
            let g = (pixel[1] / 16) as u32;
            let b = (pixel[2] / 16) as u32;
            let key = (r << 8) | (g << 4) | b;
            colors.insert(key);
        }

        colors.len()
    }

    fn compute_avg_brightness(&self, rgba: &image::RgbaImage) -> f32 {
        let mut total = 0.0f32;
        for pixel in rgba.pixels() {
            // Luminance formula
            total += 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
        }
        total / (rgba.pixels().count() as f32 * 255.0)
    }

    fn compute_brightness_variance(&self, rgba: &image::RgbaImage) -> f32 {
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut count = 0.0f32;

        for pixel in rgba.pixels() {
            let lum = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
            sum += lum;
            sum_sq += lum * lum;
            count += 1.0;
        }

        if count < 2.0 { return 0.0; }

        let mean = sum / count;
        (sum_sq / count) - (mean * mean)
    }
}

impl Default for ImageTypeClassifier {
    fn default() -> Self {
        Self::new()
    }
}
