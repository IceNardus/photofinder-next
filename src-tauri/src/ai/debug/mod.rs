//! Debug visualization for face detection pipeline

use std::path::{Path, PathBuf};
use image::{Rgb, RgbImage, ImageBuffer};
use tracing::info;

use crate::ai::face::DetectedFace;

/// Debug configuration
#[derive(Debug, Clone)]
pub struct DebugConfig {
    pub enabled: bool,
    pub debug_dir: PathBuf,
    pub save_detection_overlay: bool,
    pub save_aligned_faces: bool,
}

impl DebugConfig {
    pub fn new(debug_dir: &Path) -> Self {
        Self {
            enabled: true,
            debug_dir: debug_dir.to_path_buf(),
            save_detection_overlay: true,
            save_aligned_faces: true,
        }
    }

    pub fn debug_dir(&self) -> &Path {
        &self.debug_dir
    }
}

/// Save detection overlay image with bounding boxes
pub fn save_detection_overlay(
    image_path: &str,
    faces: &[DetectedFace],
    config: &DebugConfig,
) -> Option<String> {
    if !config.save_detection_overlay {
        return None;
    }

    let img = match image::open(image_path) {
        Ok(img) => img,
        Err(e) => {
            info!("Failed to open image for overlay: {}", e);
            return None;
        }
    };

    let (width, height) = (img.width(), img.height());
    let mut rgb = img.to_rgb8();

    // Draw each face bbox
    for (i, face) in faces.iter().enumerate() {
        let [x1, y1, x2, y2] = face.bbox;

        // Clamp to image bounds
        let x1 = x1.max(0.0).min(width as f32) as u32;
        let y1 = y1.max(0.0).min(height as f32) as u32;
        let x2 = x2.max(0.0).min(width as f32) as u32;
        let y2 = y2.max(0.0).min(height as f32) as u32;

        // Choose color based on score
        let color = if face.score > 0.8 {
            Rgb([0, 255, 0]) // Green for high confidence
        } else if face.score > 0.6 {
            Rgb([255, 255, 0]) // Yellow for medium
        } else {
            Rgb([255, 0, 0]) // Red for low
        };

        // Draw rectangle
        draw_rectangle(&mut rgb, x1, y1, x2, y2, color);

        // Draw label background
        let label = format!("[{}] {:.2}", i, face.score);
        draw_text(&mut rgb, x1, y1.saturating_sub(20), &label, color);
    }

    // Save image
    let filename = Path::new(image_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown.jpg");

    let output_path = config.debug_dir.join("detections").join(format!("{}_detected.jpg", filename));

    // Create dir if not exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    if rgb.save(&output_path).is_ok() {
        info!("Saved detection overlay: {}", output_path.display());
        Some(output_path.to_string_lossy().to_string())
    } else {
        None
    }
}

/// Save aligned face crops for inspection
pub fn save_aligned_faces(
    aligned_faces: &[image::RgbImage],
    image_path: &str,
    config: &DebugConfig,
) -> Option<String> {
    if !config.save_aligned_faces || aligned_faces.is_empty() {
        return None;
    }

    let filename = Path::new(image_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let output_dir = config.debug_dir.join("aligned").join(filename);

    std::fs::create_dir_all(&output_dir).ok();

    for (i, face) in aligned_faces.iter().enumerate() {
        let face_path = output_dir.join(format!("{:03}.jpg", i));
        face.save(&face_path).ok();
    }

    info!("Saved {} aligned faces to {}", aligned_faces.len(), output_dir.display());
    Some(output_dir.to_string_lossy().to_string())
}

/// Save aligned face crop for debug inspection
pub fn save_aligned_face_debug(
    aligned: &image::RgbImage,
    image_path: &str,
    config: &DebugConfig,
    face_index: usize,
) -> Option<String> {
    if !config.save_aligned_faces {
        return None;
    }

    let filename = Path::new(image_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let output_dir = config.debug_dir.join("aligned").join(filename);
    std::fs::create_dir_all(&output_dir).ok();

    let face_path = output_dir.join(format!("{:03}_aligned.jpg", face_index));
    if aligned.save(&face_path).is_ok() {
        Some(face_path.to_string_lossy().to_string())
    } else {
        None
    }
}

/// Draw rectangle on image
fn draw_rectangle(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
    // Horizontal lines
    for x in x1..x2.min(img.width()) {
        if y1 < img.height() {
            img.put_pixel(x, y1, color);
        }
        if y2 < img.height() {
            img.put_pixel(x, y2, color);
        }
    }
    // Vertical lines
    for y in y1..y2.min(img.height()) {
        if x1 < img.width() {
            img.put_pixel(x1, y, color);
        }
        if x2 < img.width() {
            img.put_pixel(x2, y, color);
        }
    }
}

/// Draw text label (simple, no font)
fn draw_text(img: &mut RgbImage, x: u32, y: u32, text: &str, color: Rgb<u8>) {
    let bg_color = Rgb([0, 0, 0]);
    let char_width = 8;
    let char_height = 12;

    // Draw background
    for dy in 0..char_height.min(y) {
        for dx in 0..(text.len() as u32 * char_width) {
            let px = x + dx;
            let py = y.saturating_sub(char_height) + dy;
            if px < img.width() && py < img.height() {
                img.put_pixel(px, py, bg_color);
            }
        }
    }

    // For now just draw a simple marker since we don't have font
    // In production you'd use imageproc or similar
    let _ = (text.len(), char_width, char_height);
    let _ = color;
}

/// NMS statistics for debugging
#[derive(Debug)]
pub struct NmsStats {
    pub raw_count: usize,
    pub after_nms_count: usize,
    pub iou_threshold: f32,
}

impl NmsStats {
    pub fn new(raw: usize, after: usize, iou_threshold: f32) -> Self {
        Self {
            raw_count: raw,
            after_nms_count: after,
            iou_threshold,
        }
    }

    pub fn log(&self) {
        let reduction = if self.raw_count > 0 {
            (1.0 - self.after_nms_count as f32 / self.raw_count as f32) * 100.0
        } else {
            0.0
        };
        info!(
            "NMS stats: raw={}, after_nms={} (removed {:.1}%, threshold={:.2})",
            self.raw_count, self.after_nms_count, reduction, self.iou_threshold
        );
    }
}

/// Duplicate detection report
#[derive(Debug)]
pub struct DuplicateReport {
    pub high_iou_pairs: Vec<(usize, usize, f32)>, // (i, j, iou)
}

impl DuplicateReport {
    pub fn detect(faces: &[DetectedFace], iou_threshold: f32) -> Self {
        let mut high_iou_pairs = Vec::new();

        for i in 0..faces.len() {
            for j in (i + 1)..faces.len() {
                let iou = compute_iou(&faces[i].bbox, &faces[j].bbox);
                if iou > iou_threshold {
                    high_iou_pairs.push((i, j, iou));
                }
            }
        }

        if !high_iou_pairs.is_empty() {
            info!("Duplicate detection report: {} pairs with IoU > {:.2}", high_iou_pairs.len(), iou_threshold);
            for (i, j, iou) in &high_iou_pairs[..high_iou_pairs.len().min(10)] {
                info!("  bbox[{}] overlaps bbox[{}] = {:.3}", i, j, iou);
            }
        }

        Self { high_iou_pairs }
    }
}

/// Compute IoU between two bboxes
fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union_area = area_a + area_b - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}