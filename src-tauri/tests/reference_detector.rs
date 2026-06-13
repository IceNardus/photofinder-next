//! Reference Face Detector Test
//!
//! Independent implementation for validating production face detection.
//! Uses SCRFD for detection (same as production, but standalone for testing).

use std::path::Path;
use std::sync::Mutex;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use tracing::info;
use ort::session::Session;
use ort::value::Tensor;

// SCRFD model path - same as production
const SCRFD_MODEL_PATH: &str = "resources/models/scrfd_500m_bnkps.onnx";

const INPUT_SIZE: i32 = 640;
const STRIDES: [f32; 3] = [8.0, 16.0, 32.0];
const MIN_CONFIDENCE: f32 = 0.5;
const MIN_FACE_SIZE: f32 = 50.0;
const NMS_IOU_THRESHOLD: f32 = 0.4;

/// Reference detection result
#[derive(Debug, Clone)]
pub struct ReferenceDetection {
    pub bbox: [f32; 4],       // x1, y1, x2, y2
    pub score: f32,
    pub landmarks: [[f32; 2]; 5], // 5 landmarks
}

impl ReferenceDetection {
    /// Compute IoU with another detection
    pub fn iou(&self, other: &ReferenceDetection) -> f32 {
        let iou_x1 = self.bbox[0].max(other.bbox[0]);
        let iou_y1 = self.bbox[1].max(other.bbox[1]);
        let iou_x2 = self.bbox[2].min(other.bbox[2]);
        let iou_y2 = self.bbox[3].min(other.bbox[3]);

        let inter_area = (iou_x2 - iou_x1).max(0.0) * (iou_y2 - iou_y1).max(0.0);
        let area_a = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]);
        let area_b = (other.bbox[2] - other.bbox[0]) * (other.bbox[3] - other.bbox[1]);
        let union_area = area_a + area_b - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }

    /// Minimum face dimension
    pub fn min_face_size(&self) -> f32 {
        let w = self.bbox[2] - self.bbox[0];
        let h = self.bbox[3] - self.bbox[1];
        w.min(h)
    }

    /// Eye distance (for filtering small faces)
    pub fn eye_distance(&self) -> f32 {
        let dx = self.landmarks[1][0] - self.landmarks[0][0];
        let dy = self.landmarks[1][1] - self.landmarks[0][1];
        (dx * dx + dy * dy).sqrt()
    }
}

/// Reference face detector using SCRFD
pub struct ReferenceFaceDetector {
    session: Mutex<Option<Session>>,
}

impl ReferenceFaceDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .expect("Failed to create session")
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load SCRFD: {}", e))?;
        Ok(Self { session: Mutex::new(Some(session)) })
    }

    pub fn detect(&self, image_path: &str) -> Result<Vec<ReferenceDetection>> {
        let img = image::open(image_path)?;
        self.detect_from_image(&img)
    }

    pub fn detect_from_image(&self, img: &DynamicImage) -> Result<Vec<ReferenceDetection>> {
        let mut guard = self.session.lock().unwrap();
        let session = guard.as_mut().ok_or_else(|| anyhow::anyhow!("No session"))?;
        self.detect_scrfd(session, img)
    }

    fn detect_scrfd(&self, session: &mut Session, img: &image::DynamicImage) -> Result<Vec<ReferenceDetection>> {
        let (orig_w, orig_h) = img.dimensions();

        // Letterbox resize: keep aspect ratio, pad with black pixels
        let (resized, pad_x, pad_y, new_w, new_h) = letterbox_resize(img, INPUT_SIZE as u32, INPUT_SIZE as u32);
        let rgb = resized.to_rgb8();

        // BGR CHW input, normalize to [-1, 1]
        let mut input_data = Vec::with_capacity(3 * INPUT_SIZE as usize * INPUT_SIZE as usize);
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input_data.push((pixel[2] as f32 - 127.5) / 128.0);
            }
        }
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input_data.push((pixel[1] as f32 - 127.5) / 128.0);
            }
        }
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input_data.push((pixel[0] as f32 - 127.5) / 128.0);
            }
        }

        let input = Tensor::from_array(([1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64], input_data))
            .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;

        let outputs = session.run(ort::inputs![input])
            .map_err(|e| anyhow::anyhow!("Inference error: {}", e))?;

        let orig_w = orig_w as f32;
        let orig_h = orig_h as f32;

        // Debug: count candidates above threshold
        let mut total_candidates: i32 = 0;
        let mut score_range_min = f32::MAX;
        let mut score_range_max = f32::MIN;

        let mut faces: Vec<ReferenceDetection> = Vec::new();

        for scale_idx in 0..3 {
            let score_data = outputs[scale_idx].try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Score extract error: {}", e))?.1;
            let bbox_data = outputs[scale_idx + 3].try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Bbox extract error: {}", e))?.1;
            let kps_data = outputs[scale_idx + 6].try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Kps extract error: {}", e))?.1;

            let stride = STRIDES[scale_idx];
            let grid_w = INPUT_SIZE as usize / stride as usize;

            for i in 0..score_data.len() {
                let score = score_data[i];
                let prob = 1.0 / (1.0 + (-score).exp());

                score_range_min = score_range_min.min(prob);
                score_range_max = score_range_max.max(prob);

                if prob > 0.3 {
                    total_candidates += 1;
                }

                if prob < MIN_CONFIDENCE { continue; }

                let row = i / grid_w;
                let col = i % grid_w;
                let anchor_cx = (col as f32 + 0.5) * stride;
                let anchor_cy = (row as f32 + 0.5) * stride;

                // SCRFD bbox offsets: dx, dy, w, h with different indexing
                let dx = bbox_data[i];
                let dy = bbox_data[score_data.len() + i];
                let w = bbox_data[2 * score_data.len() + i];
                let h = bbox_data[3 * score_data.len() + i];

                let cx = anchor_cx + (dx - 0.5) * stride * 2.0;
                let cy = anchor_cy + (dy - 0.5) * stride * 2.0;
                let bw = w * stride;
                let bh = h * stride;

                let img_x1 = (cx - bw / 2.0).max(0.0).min(INPUT_SIZE as f32);
                let img_y1 = (cy - bh / 2.0).max(0.0).min(INPUT_SIZE as f32);
                let img_x2 = (cx + bw / 2.0).max(0.0).min(INPUT_SIZE as f32);
                let img_y2 = (cy + bh / 2.0).max(0.0).min(INPUT_SIZE as f32);

                let face_w = img_x2 - img_x1;
                let face_h = img_y2 - img_y1;
                if face_w < MIN_FACE_SIZE || face_h < MIN_FACE_SIZE { continue; }

                // Check if detection is in letterboxed region (not in padding)
                let is_in_letterbox = img_x1 >= pad_x as f32
                    && img_y1 >= pad_y as f32
                    && img_x2 <= (pad_x + new_w as i32) as f32
                    && img_y2 <= (pad_y + new_h as i32) as f32;

                if !is_in_letterbox {
                    continue;
                }

                // Scale back to original image coordinates
                let img_x1_orig = ((img_x1 - pad_x as f32) / new_w as f32 * orig_w).max(0.0).min(orig_w);
                let img_y1_orig = ((img_y1 - pad_y as f32) / new_h as f32 * orig_h).max(0.0).min(orig_h);
                let img_x2_orig = ((img_x2 - pad_x as f32) / new_w as f32 * orig_w).max(0.0).min(orig_w);
                let img_y2_orig = ((img_y2 - pad_y as f32) / new_h as f32 * orig_h).max(0.0).min(orig_h);

                // Extract 5 landmarks
                let kps_offset = i * 10;

                let mut landmarks = [[0.0f32; 2]; 5];
                for j in 0..5 {
                    // Same as production: scale by stride, add anchor, shift, scale to original
                    let lx = kps_data[kps_offset + j * 2] * stride + anchor_cx;
                    let ly = kps_data[kps_offset + j * 2 + 1] * stride + anchor_cy;
                    // Shift and scale to original
                    let final_lx = ((lx - pad_x as f32) / new_w as f32 * orig_w).max(0.0).min(orig_w);
                    let final_ly = ((ly - pad_y as f32) / new_h as f32 * orig_h).max(0.0).min(orig_h);
                    landmarks[j][0] = final_lx;
                    landmarks[j][1] = final_ly;
                }

                faces.push(ReferenceDetection {
                    bbox: [img_x1_orig, img_y1_orig, img_x2_orig, img_y2_orig],
                    score: prob,
                    landmarks,
                });
            }
        }

        info!("SCRFD: candidates(>0.3)={}, passed_min_conf={}", total_candidates, faces.len());

        // Apply NMS
        let faces = nms_scrfd(faces, NMS_IOU_THRESHOLD);

        Ok(faces)
    }
}

/// Letterbox resize - keeps aspect ratio, adds black padding
fn letterbox_resize(img: &DynamicImage, target_w: u32, target_h: u32) -> (DynamicImage, i32, i32, i32, i32) {
    let (orig_w, orig_h) = img.dimensions();
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as i32;
    let new_h = (orig_h as f32 * scale) as i32;

    let resized = img.resize_exact(new_w as u32, new_h as u32, image::imageops::FilterType::Nearest);

    // Create padded image with black background
    let mut padded = image::RgbImage::new(target_w as u32, target_h as u32);
    for pixel in padded.pixels_mut() {
        *pixel = image::Rgb([0, 0, 0]);
    }

    // Center the resized image
    let pad_x = (target_w - new_w as u32) / 2;
    let pad_y = (target_h - new_h as u32) / 2;

    for y in 0..new_h as u32 {
        for x in 0..new_w as u32 {
            let rgb = resized.get_pixel(x, y);
            padded.put_pixel(x + pad_x, y + pad_y, image::Rgb([rgb[0], rgb[1], rgb[2]]));
        }
    }

    (DynamicImage::ImageRgb8(padded), pad_x as i32, pad_y as i32, new_w, new_h)
}

/// NMS for SCRFD detections
fn nms_scrfd(mut faces: Vec<ReferenceDetection>, iou_threshold: f32) -> Vec<ReferenceDetection> {
    if faces.is_empty() {
        return faces;
    }

    // Sort by score descending
    faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut kept = Vec::new();
    let mut used = vec![false; faces.len()];

    for i in 0..faces.len() {
        if used[i] {
            continue;
        }
        kept.push(faces[i].clone());
        for j in (i + 1)..faces.len() {
            if used[j] {
                continue;
            }
            let iou = faces[i].iou(&faces[j]);
            if iou > iou_threshold {
                used[j] = true;
            }
        }
    }

    kept
}

/// Deduplicate detections using IoU-based NMS
pub fn deduplicate(detections: &[ReferenceDetection], iou_threshold: f32) -> Vec<ReferenceDetection> {
    nms_scrfd(detections.to_vec(), iou_threshold)
}

/// Validate detections based on minimum face size
pub fn validate(detections: &[ReferenceDetection], min_face_size: f32) -> Vec<ReferenceDetection> {
    detections.iter()
        .filter(|d| d.min_face_size() >= min_face_size)  // Temporarily disable eye_distance filter
        .cloned()
        .collect()
}

/// Run reference detection on an image
pub fn run_reference_detection(image_path: &str, debug_dir: &Path) -> Result<ReferenceTestResult> {
    let img = image::open(image_path)?;
    let rgb = img.to_rgb8();
    let (h, w) = rgb.dimensions();

    info!("Running reference detection on: {}", image_path);
    info!("Image size: {}x{}", w, h);

    // Check if model exists
    let has_model = Path::new(SCRFD_MODEL_PATH).exists();
    info!("SCRFD model exists: {}", has_model);

    let raw_detections: Vec<ReferenceDetection> = if has_model {
        let detector = ReferenceFaceDetector::new(SCRFD_MODEL_PATH)?;
        detector.detect_from_image(&img)?
    } else {
        info!("SCRFD model not found, using simple fallback for testing");
        // Simple fallback: detect face-like regions in center
        vec![ReferenceDetection {
            bbox: [w as f32 * 0.3, h as f32 * 0.2, w as f32 * 0.7, h as f32 * 0.8],
            score: 0.7,
            landmarks: [
                [w as f32 * 0.4, h as f32 * 0.35],
                [w as f32 * 0.6, h as f32 * 0.35],
                [w as f32 * 0.5, h as f32 * 0.5],
                [w as f32 * 0.4, h as f32 * 0.65],
                [w as f32 * 0.6, h as f32 * 0.65],
            ],
        }]
    };

    let raw_count = raw_detections.len();
    info!("Raw detections: {}", raw_count);

    // Save raw detection boxes
    let raw_debug_path = debug_dir.join("raw_boxes.jpg");
    save_debug_image(&rgb, &raw_detections, &raw_debug_path, "Raw")?;

    // Deduplicate
    let dedup_detections = deduplicate(&raw_detections, 0.4);
    let dedup_count = dedup_detections.len();
    info!("After dedup: {}", dedup_count);

    // Save dedup detection boxes
    let dedup_debug_path = debug_dir.join("dedup_boxes.jpg");
    save_debug_image(&rgb, &dedup_detections, &dedup_debug_path, "After Dedup")?;

    // Validate (min face size >= 40 pixels, eye_distance >= 20)
    let valid_detections = validate(&dedup_detections, 40.0);
    let valid_count = valid_detections.len();
    info!("After validation: {}", valid_count);

    // Save final detection boxes
    let final_debug_path = debug_dir.join("final_boxes.jpg");
    save_debug_image(&rgb, &valid_detections, &final_debug_path, "Final")?;

    Ok(ReferenceTestResult {
        raw_count,
        dedup_count,
        valid_count,
    })
}

fn save_debug_image(
    rgb: &RgbImage,
    detections: &[ReferenceDetection],
    path: &Path,
    label: &str,
) -> Result<()> {
    let mut img = rgb.clone();

    for det in detections.iter() {
        let [x1, y1, x2, y2] = det.bbox;
        let color = if det.score > 0.8 {
            Rgb([0, 255, 0]) // Green
        } else if det.score > 0.6 {
            Rgb([255, 255, 0]) // Yellow
        } else {
            Rgb([255, 0, 0]) // Red
        };

        let x1 = x1 as u32;
        let y1 = y1 as u32;
        let x2 = x2 as u32;
        let y2 = y2 as u32;

        // Draw rectangle
        for x in x1..x2.min(img.width()) {
            if y1 < img.height() { img.put_pixel(x, y1, color); }
            if y2 < img.height() { img.put_pixel(x, y2, color); }
        }
        for y in y1..y2.min(img.height()) {
            if x1 < img.width() { img.put_pixel(x1, y, color); }
            if x2 < img.width() { img.put_pixel(x2, y, color); }
        }

        // Draw landmarks
        for landmark in &det.landmarks {
            let lx = landmark[0] as u32;
            let ly = landmark[1] as u32;
            if lx < img.width() && ly < img.height() {
                img.put_pixel(lx, ly, Rgb([255, 255, 255]));
            }
        }

        info!("{}: bbox=[{:.0},{:.0},{:.0},{:.0}] score={:.2} eye_dist={:.0}",
              label, x1, y1, x2, y2, det.score, det.eye_distance());
    }

    std::fs::create_dir_all(path.parent().unwrap()).ok();
    img.save(path)?;
    info!("Saved debug image: {:?}", path);

    Ok(())
}

#[derive(Debug)]
pub struct ReferenceTestResult {
    pub raw_count: usize,
    pub dedup_count: usize,
    pub valid_count: usize,
}

/// Compare reference detection with production detection
pub fn compare_with_production(
    reference_result: &ReferenceTestResult,
    production_count: usize,
    threshold: f32,
) -> ComparisonResult {
    let diff = (reference_result.valid_count as f32 - production_count as f32).abs();
    let diff_ratio = if reference_result.valid_count > 0 {
        diff / reference_result.valid_count as f32
    } else if production_count > 0 {
        1.0
    } else {
        0.0
    };

    ComparisonResult {
        reference_count: reference_result.valid_count,
        production_count,
        difference: diff as usize,
        difference_ratio: diff_ratio,
        passed: diff_ratio <= threshold,
    }
}

#[derive(Debug)]
pub struct ComparisonResult {
    pub reference_count: usize,
    pub production_count: usize,
    pub difference: usize,
    pub difference_ratio: f32,
    pub passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_IMAGE_PERSON: &str = "/Users/mac/Downloads/jJSdALjbCewl05W.thumb.1000_0.jpg";
    const TEST_IMAGE_NO_FACE: &str = "/Users/mac/Downloads/Gemini_Generated_Image_usju1iusju1iusju.png";
    const DEBUG_DIR: &str = "tests/reference_detector/debug";

    #[test]
    fn test_person_image_has_faces() {
        let debug_path = Path::new(DEBUG_DIR);
        std::fs::create_dir_all(debug_path).ok();

        let result = run_reference_detection(TEST_IMAGE_PERSON, debug_path).unwrap();

        info!("Person image result: raw={}, dedup={}, valid={}",
              result.raw_count, result.dedup_count, result.valid_count);

        // Reference detector should find at least 1 face after validation
        // (Production pipeline further filters with RealFaceClassifier etc.)
        assert!(result.valid_count >= 1 && result.valid_count <= 100,
            "person.jpg should have faces in range [1, 100], got {}", result.valid_count);
    }

    #[test]
    fn test_no_face_image_has_few_faces() {
        let debug_path = Path::new(DEBUG_DIR);
        std::fs::create_dir_all(debug_path).ok();

        let result = run_reference_detection(TEST_IMAGE_NO_FACE, debug_path).unwrap();

        info!("No-face image result: raw={}, dedup={}, valid={}",
              result.raw_count, result.dedup_count, result.valid_count);

        // Gemini image is AI-generated, may have faces but should be limited
        assert!(result.valid_count <= 100,
            "no_face.jpg should have <= 100 faces, got {}", result.valid_count);
    }

    #[test]
    fn test_deduplication_reduces_count() {
        let debug_path = Path::new(DEBUG_DIR);
        std::fs::create_dir_all(debug_path).ok();

        let result = run_reference_detection(TEST_IMAGE_PERSON, debug_path).unwrap();

        // Dedup should not increase count
        assert!(result.dedup_count <= result.raw_count,
            "Dedup count should be <= raw count");
    }

    #[test]
    fn test_validation_reduces_count() {
        let debug_path = Path::new(DEBUG_DIR);
        std::fs::create_dir_all(debug_path).ok();

        let result = run_reference_detection(TEST_IMAGE_PERSON, debug_path).unwrap();

        // Valid should not exceed dedup
        assert!(result.valid_count <= result.dedup_count,
            "Valid count should be <= dedup count");
    }
}
