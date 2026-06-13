//! YOLOv8-based object detection

use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};
use tracing::info;

/// Bounding box in pixel coordinates
#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

const NMS_IOU_THRESHOLD: f32 = 0.4;

impl BBox {
    /// Compute area of the bounding box
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    /// Compute IoU (Intersection over Union) with another bbox
    pub fn iou(&self, other: &BBox) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);
        let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
        let area_a = self.area();
        let area_b = other.area();
        let union_area = area_a + area_b - inter_area;
        if union_area <= 0.0 { return 0.0; }
        inter_area / union_area
    }

    /// Crop image using this bounding box
    pub fn crop_image(&self, img: &RgbImage) -> RgbImage {
        let x1 = self.x1 as u32;
        let y1 = self.y1 as u32;
        let x2 = self.x2.min(img.width() as f32) as u32;
        let y2 = self.y2.min(img.height() as f32) as u32;

        if x2 <= x1 || y2 <= y1 {
            return RgbImage::new(1, 1);
        }

        image::imageops::crop_imm(img, x1, y1, x2 - x1, y2 - y1).to_image()
    }
}

/// Detected object with bounding box and confidence
#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub bbox: BBox,
    pub class_id: i32,
    pub class_name: String,
    pub confidence: f32,
}

/// YOLOv8 object detector
pub struct YoloV8Detector {
    session: Arc<Mutex<Option<Session>>>,
    input_width: u32,
    input_height: u32,
    input_name: String,
    output_names: Vec<String>,
    class_names: Vec<String>,
}

unsafe impl Send for YoloV8Detector {}
unsafe impl Sync for YoloV8Detector {}

impl YoloV8Detector {
    /// Create a new YOLOv8 detector from an ONNX model file
    pub fn new(model_path: &str) -> Result<Self, String> {
        let session = Session::builder()
            .expect("Failed to create session")
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load YOLOv8 model: {}", e))?;

        let inputs = session.inputs();
        let outputs = session.outputs();

        let input_name = inputs.first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());

        let output_names: Vec<String> = outputs.iter()
            .map(|o| o.name().to_string())
            .collect();

        // YOLOv8 model input size is 640x640
        let (input_width, input_height) = (640, 640);

        // COCO class names
        let class_names: Vec<String> = vec![
            "person".to_string(), "bicycle".to_string(), "car".to_string(), "motorcycle".to_string(),
            "airplane".to_string(), "bus".to_string(), "train".to_string(), "truck".to_string(),
            "boat".to_string(), "traffic light".to_string(), "fire hydrant".to_string(),
            "stop sign".to_string(), "parking meter".to_string(), "bench".to_string(),
            "bird".to_string(), "cat".to_string(), "dog".to_string(), "horse".to_string(),
            "sheep".to_string(), "cow".to_string(), "elephant".to_string(), "bear".to_string(),
            "zebra".to_string(), "giraffe".to_string(), "backpack".to_string(),
            "umbrella".to_string(), "handbag".to_string(), "tie".to_string(),
            "suitcase".to_string(), "frisbee".to_string(), "skis".to_string(),
            "snowboard".to_string(), "sports ball".to_string(), "kite".to_string(),
            "baseball bat".to_string(), "baseball glove".to_string(), "skateboard".to_string(),
            "surfboard".to_string(), "tennis racket".to_string(), "bottle".to_string(),
            "wine glass".to_string(), "cup".to_string(), "fork".to_string(), "knife".to_string(),
            "spoon".to_string(), "bowl".to_string(), "banana".to_string(), "apple".to_string(),
            "sandwich".to_string(), "orange".to_string(), "broccoli".to_string(),
            "carrot".to_string(), "hot dog".to_string(), "pizza".to_string(), "donut".to_string(),
            "cake".to_string(), "chair".to_string(), "couch".to_string(),
            "potted plant".to_string(), "bed".to_string(), "dining table".to_string(),
            "toilet".to_string(), "tv".to_string(), "laptop".to_string(), "mouse".to_string(),
            "remote".to_string(), "keyboard".to_string(), "cell phone".to_string(),
            "microwave".to_string(), "oven".to_string(), "toaster".to_string(),
            "sink".to_string(), "refrigerator".to_string(), "book".to_string(),
            "clock".to_string(), "vase".to_string(), "scissors".to_string(),
            "teddy bear".to_string(), "hair drier".to_string(), "toothbrush".to_string(),
        ];

        Ok(Self {
            session: Arc::new(Mutex::new(Some(session))),
            input_width,
            input_height,
            input_name,
            output_names,
            class_names,
        })
    }

    /// Detect objects in an image
    pub fn detect(&self, img: &RgbImage) -> Result<Vec<DetectedObject>, String> {
        let mut guard = self.session.lock().unwrap();
        let session = guard.as_mut().ok_or("No session")?;

        // Preprocess: resize to model input size
        let resized = image::imageops::resize(
            img,
            self.input_width,
            self.input_height,
            FilterType::Triangle,
        );

        // Convert to CHW format and normalize to [0, 1]
        let mut input_data = vec![0.0f32; 3 * self.input_width as usize * self.input_height as usize];

        for (i, pixel) in resized.pixels().enumerate() {
            input_data[i] = pixel[0] as f32 / 255.0;
            input_data[i + (self.input_width * self.input_height) as usize] = pixel[1] as f32 / 255.0;
            input_data[i + 2 * (self.input_width * self.input_height) as usize] = pixel[2] as f32 / 255.0;
        }

        // Create input tensor [1, 3, H, W]
        let shape = [1_i64, 3, self.input_height as i64, self.input_width as i64];
        let input = Tensor::from_array((shape, input_data))
            .map_err(|e| format!("Tensor error: {}", e))?;

        // Run inference
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => input])
            .map_err(|e| format!("YOLOv8 inference failed: {}", e))?;

        // Get output tensor
        let output_data = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract output: {}", e))?.1;

        let num_classes = 80;
        let num_boxes = output_data.len() / (4 + num_classes);

        info!("[YOLO] Output len={}, num_boxes={}", output_data.len(), num_boxes);

        // Check first 20 values for debugging (raw output)
        let first_20: Vec<f32> = output_data.iter().take(20).cloned().collect();
        info!("[YOLO] First 20 values: {:?}", first_20);

        // Print class scores for first box (assuming [batch, num_boxes, 84] format)
        if num_boxes > 0 {
            let first_box_scores: Vec<f32> = (0..10).map(|j| output_data[4 + j]).collect();
            info!("[YOLO] First box class scores (first 10): {:?}", first_box_scores);
            let max_conf = (0..num_classes).map(|j| output_data[4 + j]).fold(0.0f32, |a, b| a.max(b));
            info!("[YOLO] First box max class score: {}", max_conf);
        }

        let img_width = img.width() as f32;
        let img_height = img.height() as f32;

        // Scale factors
        let scale_x = img_width / self.input_width as f32;
        let scale_y = img_height / self.input_height as f32;

        let mut objects = Vec::new();

        // Try [batch, num_boxes, 84] format (row-major)
        // For box i: bbox at [i*84 .. i*84+4], classes at [i*84+4 .. i*84+84]
        for i in 0..num_boxes {
            let bbox_idx = i * 84;
            let cx = output_data[bbox_idx];
            let cy = output_data[bbox_idx + 1];
            let w = output_data[bbox_idx + 2];
            let h = output_data[bbox_idx + 3];

            // Find max class probability
            let mut max_conf = 0.0f32;
            let mut max_class = 0i32;

            for j in 0..num_classes {
                let conf = output_data[bbox_idx + 4 + j];
                if conf > max_conf {
                    max_conf = conf;
                    max_class = j as i32;
                }
            }

            // Filter by confidence threshold (raised to 0.4 to reduce noise)
            if max_conf < 0.25 {
                continue;
            }

            // Convert from center format to corner format and scale
            let x1 = ((cx - w / 2.0) * scale_x).max(0.0).min(img_width);
            let y1 = ((cy - h / 2.0) * scale_y).max(0.0).min(img_height);
            let x2 = ((cx + w / 2.0) * scale_x).max(0.0).min(img_width);
            let y2 = ((cy + h / 2.0) * scale_y).max(0.0).min(img_height);

            // Skip small boxes
            if x2 - x1 < 20.0 || y2 - y1 < 20.0 {
                continue;
            }

            let class_name = if max_class < self.class_names.len() as i32 {
                self.class_names[max_class as usize].clone()
            } else {
                format!("class_{}", max_class)
            };

            objects.push(DetectedObject {
                bbox: BBox { x1, y1, x2, y2 },
                class_id: max_class,
                class_name,
                confidence: max_conf,
            });
        }

        // Apply NMS (Non-Maximum Suppression)
        let before_nms = objects.len();
        objects.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut used = vec![false; objects.len()];

        for i in 0..objects.len() {
            if used[i] { continue; }
            keep.push(i);
            for j in (i + 1)..objects.len() {
                if used[j] { continue; }
                let iou = objects[i].bbox.iou(&objects[j].bbox);
                if iou > NMS_IOU_THRESHOLD { used[j] = true; }
            }
        }

        let after_nms = keep.len();
        if before_nms > 0 {
            info!("YOLO detect: before_nms={}, after_nms={} (removed {:.1}%, threshold={:.2})",
                before_nms, after_nms,
                (1.0 - after_nms as f32 / before_nms as f32) * 100.0,
                NMS_IOU_THRESHOLD);
        }

        objects = keep.into_iter().map(|i| objects[i].clone()).collect();
        Ok(objects)
    }

    /// Detect and crop objects from an image, returning the cropped patches
    pub fn detect_and_crop(&self, img: &RgbImage) -> Result<Vec<(DetectedObject, RgbImage)>, String> {
        let objects = self.detect(img)?;

        let mut results = Vec::new();
        for obj in objects {
            let crop = obj.bbox.crop_image(img);
            results.push((obj, crop));
        }

        Ok(results)
    }
}