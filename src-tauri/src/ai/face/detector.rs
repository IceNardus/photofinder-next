//! SCRFD Face Detector for PhotoFinder Next

use std::sync::Mutex;
use image::{GenericImageView, imageops::FilterType};
use tracing::{info, warn};
use ort::session::Session;
use ort::value::Tensor;

const INPUT_SIZE: i32 = 640;
const STRIDES: [f32; 3] = [8.0, 16.0, 32.0];
const MIN_CONFIDENCE: f32 = 0.3;   // Lower threshold for probability
const MIN_RAW_SCORE: f32 = 0.05;   // Very low raw score threshold
const MIN_FACE_W: f32 = 24.0;      // Minimum face width in original image pixels
const MIN_FACE_H: f32 = 24.0;      // Minimum face height in original image pixels
const NMS_IOU_THRESHOLD: f32 = 0.4;

#[derive(Debug, Clone)]
pub struct FiveKeypoints {
    pub left_eye: (f32, f32),
    pub right_eye: (f32, f32),
    pub nose: (f32, f32),
    pub left_mouth: (f32, f32),
    pub right_mouth: (f32, f32),
}

impl FiveKeypoints {
    pub fn from_scrfd(kps_data: &[f32], anchor_cx: f32, anchor_cy: f32, stride: f32) -> Self {
        let scale = |v: f32| v * stride;
        Self {
            left_eye: (scale(kps_data[0]) + anchor_cx, scale(kps_data[1]) + anchor_cy),
            right_eye: (scale(kps_data[2]) + anchor_cx, scale(kps_data[3]) + anchor_cy),
            nose: (scale(kps_data[4]) + anchor_cx, scale(kps_data[5]) + anchor_cy),
            left_mouth: (scale(kps_data[6]) + anchor_cx, scale(kps_data[7]) + anchor_cy),
            right_mouth: (scale(kps_data[8]) + anchor_cx, scale(kps_data[9]) + anchor_cy),
        }
    }

    pub fn scale_to_image(&self, scale_x: f32, scale_y: f32) -> FiveKeypoints {
        FiveKeypoints {
            left_eye: (self.left_eye.0 * scale_x, self.left_eye.1 * scale_y),
            right_eye: (self.right_eye.0 * scale_x, self.right_eye.1 * scale_y),
            nose: (self.nose.0 * scale_x, self.nose.1 * scale_y),
            left_mouth: (self.left_mouth.0 * scale_x, self.left_mouth.1 * scale_y),
            right_mouth: (self.right_mouth.0 * scale_x, self.right_mouth.1 * scale_y),
        }
    }

    pub fn shift(&self, dx: f32, dy: f32) -> FiveKeypoints {
        FiveKeypoints {
            left_eye: (self.left_eye.0 - dx, self.left_eye.1 - dy),
            right_eye: (self.right_eye.0 - dx, self.right_eye.1 - dy),
            nose: (self.nose.0 - dx, self.nose.1 - dy),
            left_mouth: (self.left_mouth.0 - dx, self.left_mouth.1 - dy),
            right_mouth: (self.right_mouth.0 - dx, self.right_mouth.1 - dy),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaceBBox {
    pub x1: f32, pub y1: f32,
    pub x2: f32, pub y2: f32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: [f32; 4],
    pub score: f32,
    pub keypoints: FiveKeypoints,
}

pub struct FaceDetector {
    session: Mutex<Option<Session>>,
}

// Safety: FaceDetector wraps Session in Mutex, which ensures thread-safe access
unsafe impl Send for FaceDetector {}
unsafe impl Sync for FaceDetector {}

impl FaceDetector {
    pub fn new(model_path: &str) -> Result<Self, String> {
        info!("Loading SCRFD from: {}", model_path);
        let session = Session::builder()
            .expect("Failed to create session")
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load SCRFD: {}", e))?;

        Ok(Self { session: Mutex::new(Some(session)) })
    }

    pub fn detect(&self, image_path: &str) -> Result<Vec<DetectedFace>, String> {
        let img = image::open(image_path).map_err(|e| format!("Failed to open image: {}", e))?;
        self.detect_from_image(&img)
    }

    pub fn detect_from_image(&self, img: &image::DynamicImage) -> Result<Vec<DetectedFace>, String> {
        let mut guard = self.session.lock().unwrap();
        let session = guard.as_mut().ok_or("No session")?;
        self.detect_scrfd(session, img)
    }

    fn detect_scrfd(&self, session: &mut Session, img: &image::DynamicImage) -> Result<Vec<DetectedFace>, String> {
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
            .map_err(|e| format!("Tensor error: {}", e))?;

        let outputs = session.run(ort::inputs![input])
            .map_err(|e| format!("Inference error: {}", e))?;

        // DEBUG: Print all output shapes
        eprintln!("\n=== SCRFD OUTPUT SHAPES ===");
        for (i, output) in outputs.iter().enumerate() {
            let shape = output.1.shape();
            eprintln!("output{} shape: {:?}", i, shape);
        }
        eprintln!("===========================\n");

        let mut faces: Vec<(FaceBBox, FiveKeypoints)> = Vec::new();

        for scale_idx in 0..3 {
            let score_data = outputs[scale_idx].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1;
            let bbox_data = outputs[scale_idx + 3].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1;
            let kps_data = outputs[scale_idx + 6].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1;

            // DEBUG: Print max score per scale
            let max_score = score_data.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mean_score = score_data.iter().sum::<f32>() / score_data.len() as f32;
            eprintln!("[SCORE_STATS] scale={} max={:.4} mean={:.4} len={}", scale_idx, max_score, mean_score, score_data.len());

            // Debug: print first few raw score and bbox values
            let score_sample: String = score_data.iter().take(5)
                .map(|s| format!("{:.4}", s))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("[RAW_TENSOR] scale={} top5_scores=[{}]", scale_idx, score_sample);

            let bbox_sample: String = bbox_data.iter().take(8)
                .map(|s| format!("{:.3}", s))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("[RAW_TENSOR] scale={} bbox_first8=[{}]", scale_idx, bbox_sample);

            let kps_raw_sample: String = kps_data.iter().take(10)
                .map(|s| format!("{:.3}", s))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("[RAW_TENSOR] scale={} kps_first10=[{}]", scale_idx, kps_raw_sample);

            let stride = STRIDES[scale_idx];
            let grid_w = INPUT_SIZE as usize / stride as usize;

            for i in 0..score_data.len() {
                let raw_score = score_data[i];
                // Treat raw_score as probability directly (model already outputs sigmoid)
                if raw_score < 0.05 { continue; }  // Lower threshold
                let prob = raw_score;  // No sigmoid - model already outputs probability
                if prob < 0.3 { continue; }  // Lower confidence threshold

                // Dual anchor indexing: 12800 = 80×80×2
                let anchor_idx = i % 2;
                let pos_idx = i / 2;
                let row = pos_idx / grid_w;
                let col = pos_idx % grid_w;
                let anchor_cx = (col as f32 + 0.5) * stride;
                let anchor_cy = (row as f32 + 0.5) * stride;

                // bbox tensor shape: [1, num_detections, 4] = [batch, num_detections, 4]
                // Layout: [det0_l, det0_t, det0_r, det0_b] - LEFT TOP RIGHT BOTTOM (official SCRFD format)
                let bbox_idx = i * 4;
                let l = bbox_data[bbox_idx] * stride;
                let t = bbox_data[bbox_idx + 1] * stride;
                let r = bbox_data[bbox_idx + 2] * stride;
                let b = bbox_data[bbox_idx + 3] * stride;

                let img_x1 = (anchor_cx - l).max(0.0).min(INPUT_SIZE as f32);
                let img_y1 = (anchor_cy - t).max(0.0).min(INPUT_SIZE as f32);
                let img_x2 = (anchor_cx + r).max(0.0).min(INPUT_SIZE as f32);
                let img_y2 = (anchor_cy + b).max(0.0).min(INPUT_SIZE as f32);

                let face_w = img_x2 - img_x1;
                let face_h = img_y2 - img_y1;
                // Use lower thresholds for canvas coordinates (scale ~2x from original)
                if face_w < 16.0 || face_h < 16.0 {
                    eprintln!("[FILTER_SIZE] scale={} raw={:.4} w={:.1} h={:.1} anchor=({},{})",
                        scale_idx, raw_score, face_w, face_h, anchor_cx, anchor_cy);
                    continue;
                }

                // Scale back to original image coordinates
                // Detection is on 640x640 letterboxed canvas at coords img_x1,img_y1
                // First check if detection is in the letterboxed region (not in padding)
                let is_in_letterbox = img_x1 >= pad_x as f32
                    && img_y1 >= pad_y as f32
                    && img_x2 <= (pad_x + new_w as i32) as f32
                    && img_y2 <= (pad_y + new_h as i32) as f32;

                if !is_in_letterbox {
                    eprintln!("[FILTER_PAD] scale={} raw={:.4} anchor=({:.1},{:.1}) pad=({},{}) bbox=({:.0},{:.0},{:.0},{:.0})",
                        scale_idx, raw_score, anchor_cx, anchor_cy, pad_x, pad_y, img_x1, img_y1, img_x2, img_y2);
                    continue; // Skip detections in padding area
                }

                let img_x1_orig = ((img_x1 - pad_x as f32) / new_w as f32 * orig_w as f32).max(0.0).min(orig_w as f32);
                let img_y1_orig = ((img_y1 - pad_y as f32) / new_h as f32 * orig_h as f32).max(0.0).min(orig_h as f32);
                let img_x2_orig = ((img_x2 - pad_x as f32) / new_w as f32 * orig_w as f32).max(0.0).min(orig_w as f32);
                let img_y2_orig = ((img_y2 - pad_y as f32) / new_h as f32 * orig_h as f32).max(0.0).min(orig_h as f32);

                // CRITICAL DEBUG: Print raw bbox values to determine format (ltrb vs xywh)
                eprintln!("[BBOX_RAW] i={} anchor=({:.1},{:.1}) stride={} raw=[{:.3},{:.3},{:.3},{:.3}] decoded=({:.0},{:.0},{:.0},{:.0})",
                    i, anchor_cx, anchor_cy, stride,
                    bbox_data[bbox_idx], bbox_data[bbox_idx+1], bbox_data[bbox_idx+2], bbox_data[bbox_idx+3],
                    img_x1, img_y1, img_x2, img_y2);

                let bbox = FaceBBox {
                    x1: img_x1_orig,
                    y1: img_y1_orig,
                    x2: img_x2_orig,
                    y2: img_y2_orig,
                    score: prob,
                };

                // KPS tensor shape: [1, num_detections, 10] = [batch, num_detections, 10]
                // Layout: [det0_kp0_x, det0_kp0_y, det0_kp1_x, ..., det1_kp0_x, ...]
                let kps_offset = i * 10;
                let kps_raw = &kps_data[kps_offset..kps_offset + 10];

                // Print raw KPS values for left eye (first 2 values)
                eprintln!("[KPS_RAW] i={} anchor=({:.1},{:.1}) stride={} raw_kps=[{:.3},{:.3},{:.3},{:.3},{:.3}]",
                    i, anchor_cx, anchor_cy, stride,
                    kps_raw[0], kps_raw[1], kps_raw[2], kps_raw[3], kps_raw[4]);

                // Compare two decoding formulas:
                // Formula A: anchor + raw * stride (current)
                // Formula B: anchor + raw * stride * 2 (like bbox)
                let decoded_a_x = anchor_cx + kps_raw[0] * stride;
                let decoded_a_y = anchor_cy + kps_raw[1] * stride;
                let decoded_b_x = anchor_cx + kps_raw[0] * stride * 2.0;
                let decoded_b_y = anchor_cy + kps_raw[1] * stride * 2.0;

                eprintln!("[KPS_COMPARE] i={} A=({:.1},{:.1}) B=({:.1},{:.1})",
                    i, decoded_a_x, decoded_a_y, decoded_b_x, decoded_b_y);

                // KPS decode: shift+scale anchor, then add raw*stride (NOT scaled)
                let kps_scale_x = orig_w as f32 / new_w as f32;
                let kps_scale_y = orig_h as f32 / new_h as f32;

                let kps = FiveKeypoints {
                    left_eye: (
                        (anchor_cx - pad_x as f32) * kps_scale_x + kps_raw[0] * stride,
                        (anchor_cy - pad_y as f32) * kps_scale_y + kps_raw[1] * stride,
                    ),
                    right_eye: (
                        (anchor_cx - pad_x as f32) * kps_scale_x + kps_raw[2] * stride,
                        (anchor_cy - pad_y as f32) * kps_scale_y + kps_raw[3] * stride,
                    ),
                    nose: (
                        (anchor_cx - pad_x as f32) * kps_scale_x + kps_raw[4] * stride,
                        (anchor_cy - pad_y as f32) * kps_scale_y + kps_raw[5] * stride,
                    ),
                    left_mouth: (
                        (anchor_cx - pad_x as f32) * kps_scale_x + kps_raw[6] * stride,
                        (anchor_cy - pad_y as f32) * kps_scale_y + kps_raw[7] * stride,
                    ),
                    right_mouth: (
                        (anchor_cx - pad_x as f32) * kps_scale_x + kps_raw[8] * stride,
                        (anchor_cy - pad_y as f32) * kps_scale_y + kps_raw[9] * stride,
                    ),
                };

                eprintln!("[KPS_TRACE] i={} anchor=({:.1},{:.1}) pad=({},{}) stride={} scale=({:.4},{:.4}) le=({:.1},{:.1})",
                    i, anchor_cx, anchor_cy, pad_x, pad_y, stride, kps_scale_x, kps_scale_y,
                    kps.left_eye.0, kps.left_eye.1);

                // KPS validation: check if keypoints fall inside bbox
                // If keypoints are far outside bbox, reduce face confidence
                let kps_in_bbox = kps.left_eye.0 >= bbox.x1 - 50.0
                    && kps.left_eye.0 <= bbox.x2 + 50.0
                    && kps.left_eye.1 >= bbox.y1 - 50.0
                    && kps.left_eye.1 <= bbox.y2 + 50.0
                    && kps.right_eye.0 >= bbox.x1 - 50.0
                    && kps.right_eye.0 <= bbox.x2 + 50.0
                    && kps.right_eye.1 >= bbox.y1 - 50.0
                    && kps.right_eye.1 <= bbox.y2 + 50.0;

                // Apply small penalty if KPS outside bbox (not deleted, just reduced)
                let adjusted_score = if kps_in_bbox {
                    bbox.score
                } else {
                    bbox.score * 0.7  // Reduce score by 30% if KPS inconsistent with bbox
                };

                let bbox_adjusted = FaceBBox {
                    x1: bbox.x1,
                    y1: bbox.y1,
                    x2: bbox.x2,
                    y2: bbox.y2,
                    score: adjusted_score,
                };

                faces.push((bbox_adjusted, kps));
            }
        }

        // NMS
        faces.sort_by(|a, b| b.0.score.partial_cmp(&a.0.score).unwrap());
        let before_nms = faces.len();

        // CRITICAL DEBUG: Print all faces going into NMS to catch coordinate swap
        eprintln!("[NMS_INPUT] count={}", faces.len());
        for (idx, f) in faces.iter().take(10).enumerate() {
            let w = f.0.x2 - f.0.x1;
            let h = f.0.y2 - f.0.y1;
            eprintln!("  [{}] bbox=({:.0},{:.0},{:.0},{:.0}) w={:.0} h={:.0} s={:.3}",
                idx, f.0.x1, f.0.y1, f.0.x2, f.0.y2, w, h, f.0.score);
        }

        // Debug: print top 10 scores and sizes
        if !faces.is_empty() {
            let top_info: String = faces.iter().take(5)
                .map(|f| {
                    let w = f.0.x2 - f.0.x1;
                    let h = f.0.y2 - f.0.y1;
                    format!("({:.0},{:.0},{:.0},{:.0},s={:.3})", f.0.x1, f.0.y1, w, h, f.0.score)
                })
                .collect::<Vec<_>>()
                .join(" | ");
            eprintln!("[SCRFD_DEBUG] before_nms={}, top5=[{}]", before_nms, top_info);
        }

        // Enhanced Soft-NMS: 结合 IoU + KPS中心距 + 尺度检查
        // Sort by score descending
        faces.sort_by(|a, b| b.0.score.partial_cmp(&a.0.score).unwrap());

        let mut scores_soft: Vec<f32> = faces.iter().map(|f| f.0.score).collect();
        let mut keep = Vec::new();

        for i in 0..faces.len() {
            if scores_soft[i] < 0.1 { continue; }  // Skip already decayed boxes

            keep.push(i);

            for j in (i + 1)..faces.len() {
                if scores_soft[j] < 0.1 { continue; }

                let a_bbox = &faces[i].0;
                let b_bbox = &faces[j].0;
                let a_kps = &faces[i].1;
                let b_kps = &faces[j].1;

                // 1. IoU 计算
                let inter_x1 = a_bbox.x1.max(b_bbox.x1);
                let inter_y1 = a_bbox.y1.max(b_bbox.y1);
                let inter_x2 = a_bbox.x2.min(b_bbox.x2);
                let inter_y2 = a_bbox.y2.min(b_bbox.y2);
                let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
                let area_a = (a_bbox.x2 - a_bbox.x1) * (a_bbox.y2 - a_bbox.y1);
                let area_b = (b_bbox.x2 - b_bbox.x1) * (b_bbox.y2 - b_bbox.y1);
                let iou = inter_area / (area_a + area_b - inter_area);

                // 2. KPS 中心距计算
                let a_center_x = (a_kps.left_eye.0 + a_kps.right_eye.0) / 2.0;
                let a_center_y = (a_kps.left_eye.1 + a_kps.right_eye.1) / 2.0;
                let b_center_x = (b_kps.left_eye.0 + b_kps.right_eye.0) / 2.0;
                let b_center_y = (b_kps.left_eye.1 + b_kps.right_eye.1) / 2.0;
                let kps_dist = ((a_center_x - b_center_x).powi(2) + (a_center_y - b_center_y).powi(2)).sqrt();

                // 3. 尺度比例检查
                let scale_a = (a_bbox.x2 - a_bbox.x1).max(a_bbox.y2 - a_bbox.y1);
                let scale_b = (b_bbox.x2 - b_bbox.x1).max(b_bbox.y2 - b_bbox.y1);
                let scale_ratio = scale_a / scale_b.max(1.0);

                // 综合判断：如果 IoU 较高 或 KPS中心很近 或 尺度接近，则是重复检测
                let is_duplicate = iou > 0.35
                    || (iou > 0.2 && kps_dist < 25.0)
                    || (iou > 0.15 && kps_dist < 15.0 && scale_ratio < 1.3);

                if is_duplicate {
                    // 更强的衰减：直接大幅降低分数
                    scores_soft[j] *= (1.0 - iou * 1.8).max(0.03);
                    if kps_dist < 15.0 {
                        scores_soft[j] = scores_soft[j].min(0.05);  // KPS很近直接压到很低
                    }
                } else if iou > 0.25 {
                    // 普通衰减
                    scores_soft[j] *= (1.0 - iou * 1.5).max(0.05);
                }
            }
        }

        let after_nms = keep.len();
        eprintln!("[NMS_ENHANCED] before={} after={} suppressed={}", before_nms, after_nms, before_nms - after_nms);

        Ok(keep.iter().map(|&i| DetectedFace {
            bbox: [faces[i].0.x1, faces[i].0.y1, faces[i].0.x2, faces[i].0.y2],
            score: faces[i].0.score,
            keypoints: faces[i].1.clone(),
        }).collect())
    }
}

/// Letterbox resize: keep aspect ratio, pad with black, return resize info
fn letterbox_resize(img: &image::DynamicImage, target_w: u32, target_h: u32) -> (image::DynamicImage, i32, i32, u32, u32) {
    let (orig_w, orig_h) = img.dimensions();
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);

    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

    // Create black canvas and paste resized image
    let mut canvas = image::RgbImage::from_pixel(target_w, target_h, image::Rgb([0, 0, 0]));
    let offset_x = ((target_w - new_w) / 2) as i32;
    let offset_y = ((target_h - new_h) / 2) as i32;

    image::imageops::overlay(&mut canvas, &resized.to_rgb8().clone(), offset_x as i64, offset_y as i64);

    (image::DynamicImage::ImageRgb8(canvas), offset_x, offset_y, new_w, new_h)
}