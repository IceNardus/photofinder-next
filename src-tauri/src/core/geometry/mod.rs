//! Geometry utilities for coordinate transformation between image spaces
//!
//! # Coordinate Spaces
//! - Original Space: The original image as loaded from disk (W x H)
//! - Processing Space: Resized image for model input (e.g., 640x640, 1024x1024)
//! - ROI Space: Intermediate space where ROIs are detected
//!
//! # Rules
//! 1. All ROI proposals are computed in Processing Space
//! 2. All crops MUST be performed in Original Space only
//! 3. Transform coordinates using scale/offset before cropping

use serde::{Deserialize, Serialize};

/// Coordinate transformation between image spaces
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Transform {
    /// Scale factor from source to destination (dst = src * scale)
    pub scale_x: f32,
    pub scale_y: f32,
    /// Offset in destination space (used for letterbox padding)
    pub offset_x: f32,
    pub offset_y: f32,
    /// Original dimensions (source of truth)
    pub original_width: u32,
    pub original_height: u32,
}

impl Transform {
    /// Create identity transform (1:1 mapping)
    pub fn identity(width: u32, height: u32) -> Self {
        Self {
            scale_x: 1.0,
            scale_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            original_width: width,
            original_height: height,
        }
    }

    /// Create transform from original to resized (for model input)
    pub fn from_resize(original_width: u32, original_height: u32, target_max: u32) -> Self {
        let scale = (target_max as f32) / (original_width.max(original_height) as f32);
        let new_w = (original_width as f32 * scale) as u32;
        let new_h = (original_height as f32 * scale) as u32;

        // For now, no offset (assumes no letterbox)
        Self {
            scale_x: new_w as f32 / original_width as f32,
            scale_y: new_h as f32 / original_height as f32,
            offset_x: 0.0,
            offset_y: 0.0,
            original_width,
            original_height,
        }
    }

    /// Transform a rectangle from processing space to original space
    pub fn to_original(&self, roi: &[f32; 4]) -> [f32; 4] {
        let x = roi[0] / self.scale_x - self.offset_x / self.scale_x;
        let y = roi[1] / self.scale_y - self.offset_y / self.scale_y;
        let w = roi[2] / self.scale_x;
        let h = roi[3] / self.scale_y;
        [x, y, w, h]
    }

    /// Transform a point from original space to processing space
    pub fn to_processing(&self, x: f32, y: f32) -> (f32, f32) {
        let px = x * self.scale_x + self.offset_x;
        let py = y * self.scale_y + self.offset_y;
        (px, py)
    }

    /// Validate that a rectangle is within original image bounds
    pub fn validate(&self, roi: &[f32; 4]) -> bool {
        let [x, y, w, h] = self.to_original(roi);
        x >= 0.0 && y >= 0.0 &&
        (x + w) as u32 <= self.original_width &&
        (y + h) as u32 <= self.original_height
    }

    /// Clamp rectangle to original image bounds
    pub fn clamp(&self, roi: &[f32; 4]) -> [f32; 4] {
        let [x, y, w, h] = self.to_original(roi);
        let x = x.max(0.0);
        let y = y.max(0.0);
        let w = w.min(self.original_width as f32 - x);
        let h = h.min(self.original_height as f32 - h);
        [x, y, w.max(1.0), h.max(1.0)]
    }
}

/// Safe crop function that validates bounds before cropping
pub fn safe_crop(
    img: &image::RgbImage,
    roi: &[f32; 4],
    name: &str,
) -> Result<image::RgbImage, String> {
    let (w, h) = img.dimensions();
    let [x, y, rw, rh] = *roi;

    // Validate bounds
    if x < 0.0 || y < 0.0 {
        eprintln!("[CROP:{}] WARNING: Negative coords clamped: {:?}", name, roi);
    }
    if (x + rw) as u32 > w || (y + rh) as u32 > h {
        eprintln!("[CROP:{}] WARNING: ROI exceeds image bounds: {:?}, img={}x{}", name, roi, w, h);
    }

    let x = x.max(0.0) as u32;
    let y = y.max(0.0) as u32;
    let rw = rw.max(1.0) as u32;
    let rh = rh.max(1.0) as u32;
    let w = w.min(x + rw);
    let h = h.min(y + rh);

    if w <= x || h <= y {
        return Err(format!("[CROP:{}] Invalid crop region: {}x{} at ({},{})", name, rw, rh, x, y));
    }

    let cropped = image::imageops::crop_imm(img, x as u32, y as u32, w - x, h - y).to_image();
    eprintln!("[CROP:{}] Cropped {}x{} from ({},{}) on {}x{} image", name, w-x, h-y, x, y, w, h);
    Ok(cropped)
}