//! Real Face Classifier - Filters out anime, CG, illustrations, animal faces, etc.

use image::RgbImage;
use tracing::info;

/// Result of real face classification
#[derive(Debug, Clone)]
pub struct RealFaceResult {
    pub is_real: bool,
    pub score: f32,
    pub reasons: Vec<String>,
}

impl RealFaceResult {
    pub fn real(score: f32) -> Self {
        Self {
            is_real: true,
            score,
            reasons: vec![],
        }
    }

    pub fn fake(score: f32, reasons: Vec<String>) -> Self {
        Self {
            is_real: false,
            score,
            reasons,
        }
    }
}

/// Classify if an aligned face is a real photo face or anime/CG/illustration/animal
pub struct RealFaceClassifier {
    min_skin_ratio: f32,
    min_edge_density: f32,
    min_face_ratio: f32,
    max_face_ratio: f32,
}

impl RealFaceClassifier {
    pub fn new() -> Self {
        Self {
            min_skin_ratio: 0.12,
            min_edge_density: 0.05,
            min_face_ratio: 0.5,
            max_face_ratio: 0.9,
        }
    }

    /// Classify an aligned 112x112 face image
    pub fn classify(&self, face: &RgbImage) -> RealFaceResult {
        let skin_ratio = self.compute_skin_ratio(face);
        let edge_density = self.compute_edge_density(face);
        let color_variance = self.compute_color_variance(face);
        let face_shape = self.compute_face_shape_ratio(face);

        let mut reasons = Vec::new();

        // Check skin ratio
        if skin_ratio < self.min_skin_ratio {
            reasons.push(format!("low_skin_ratio={:.3}", skin_ratio));
        }

        // Check edge density
        if edge_density < self.min_edge_density {
            reasons.push(format!("low_edge_density={:.3}", edge_density));
        }

        // Check color variance
        if color_variance < 0.015 {
            reasons.push(format!("low_color_variance={:.4}", color_variance));
        }

        // Check face shape ratio (humans are 0.5-0.85)
        if face_shape < self.min_face_ratio || face_shape > self.max_face_ratio {
            reasons.push(format!("non_human_face_ratio={:.3}", face_shape));
        }

        // Calculate overall score
        let score = (skin_ratio * 2.0 + edge_density * 3.0 + color_variance * 10.0).min(1.0);

        // Reject if multiple issues or score too low
        if !reasons.is_empty() {
            if reasons.len() >= 2 || score < 0.3 {
                return RealFaceResult::fake(score, reasons);
            }
            if score > 0.6 {
                return RealFaceResult::real(score);
            }
            return RealFaceResult::fake(score, reasons);
        }

        RealFaceResult::real(score)
    }

    /// Compute ratio of skin-like pixels
    pub fn compute_skin_ratio(&self, face: &RgbImage) -> f32 {
        let mut skin_count = 0u32;
        let total = (face.width() * face.height()) as f32;

        for pixel in face.pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            if self.is_skin_pixel(r, g, b) {
                skin_count += 1;
            }
        }

        skin_count as f32 / total
    }

    /// Check if a pixel looks like human skin
    fn is_skin_pixel(&self, r: f32, g: f32, b: f32) -> bool {
        let r_gt_g = r > g;
        let g_gt_b = g > b;
        let r_large = r > 95.0;
        let not_saturated = (r - g).abs() < 50.0 && (g - b).abs() < 50.0;

        let skin_hue = Self::rgb_to_hue(r, g, b);
        let is_skin_hue = skin_hue > 0.0 && skin_hue < 50.0;

        let saturation = Self::rgb_to_saturation(r, g, b);
        let not_too_light = saturation > 0.1;

        r_gt_g && r_large && not_saturated && is_skin_hue && not_too_light
    }

    /// Compute edge density using simple gradient
    pub fn compute_edge_density(&self, face: &RgbImage) -> f32 {
        let mut edge_count = 0u32;
        let w = face.width() as i32;
        let h = face.height() as i32;
        let total = (w * h) as f32;

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let center = face.get_pixel(x as u32, y as u32);
                let right = face.get_pixel((x + 1) as u32, y as u32);
                let bottom = face.get_pixel(x as u32, (y + 1) as u32);

                let grad_x = (right[0] as f32 - center[0] as f32).abs()
                    + (right[1] as f32 - center[1] as f32).abs()
                    + (right[2] as f32 - center[2] as f32).abs();
                let grad_y = (bottom[0] as f32 - center[0] as f32).abs()
                    + (bottom[1] as f32 - center[1] as f32).abs()
                    + (bottom[2] as f32 - center[2] as f32).abs();

                if grad_x + grad_y > 30.0 {
                    edge_count += 1;
                }
            }
        }

        edge_count as f32 / total
    }

    /// Compute color variance
    pub fn compute_color_variance(&self, face: &RgbImage) -> f32 {
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut sum_r2 = 0.0f32;
        let mut sum_g2 = 0.0f32;
        let mut sum_b2 = 0.0f32;
        let mut count = 0.0f32;

        for pixel in face.pixels() {
            sum_r += pixel[0] as f32;
            sum_g += pixel[1] as f32;
            sum_b += pixel[2] as f32;
            sum_r2 += (pixel[0] as f32).powi(2);
            sum_g2 += (pixel[1] as f32).powi(2);
            sum_b2 += (pixel[2] as f32).powi(2);
            count += 1.0;
        }

        if count < 1.0 {
            return 0.0;
        }

        let var_r = (sum_r2 / count) - (sum_r / count).powi(2);
        let var_g = (sum_g2 / count) - (sum_g / count).powi(2);
        let var_b = (sum_b2 / count) - (sum_b / count).powi(2);

        (var_r + var_g + var_b) / (3.0 * 255.0 * 255.0)
    }

    fn rgb_to_hue(r: f32, g: f32, b: f32) -> f32 {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        if delta == 0.0 {
            return 0.0;
        }

        let hue = if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };

        if hue < 0.0 {
            hue + 360.0
        } else {
            hue
        }
    }

    fn rgb_to_saturation(r: f32, g: f32, b: f32) -> f32 {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) / 2.0 / 255.0;

        if l == 0.0 || l == 1.0 {
            return 0.0;
        }

        let delta = max - min;
        (delta / 255.0) / (1.0 - (2.0 * l - 1.0).abs())
    }

    /// Compute face width/height ratio - humans have roughly 0.5-0.85
    fn compute_face_shape_ratio(&self, face: &RgbImage) -> f32 {
        let w = face.width() as f32;
        let _h = face.height() as f32;
        w / _h // Simplified: face is already 112x112 square
    }
}

impl Default for RealFaceClassifier {
    fn default() -> Self {
        Self::new()
    }
}
