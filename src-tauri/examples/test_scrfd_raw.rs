//! Test SCRFD raw model output - Detailed Debug Version
use ort::session::Session;
use ort::value::Tensor;
use image::GenericImageView;
use image::imageops::FilterType;

const INPUT_SIZE: i32 = 640;
const STRIDES: [f32; 3] = [8.0, 16.0, 32.0];

fn find_model(name: &str) -> String {
    let candidates = vec![
        std::path::PathBuf::from("resources/models").join(name),
        std::path::PathBuf::from("/Users/mac/ai-project/photofinder-ai/src-tauri/resources/models").join(name),
    ];
    for c in &candidates {
        if c.exists() {
            return c.to_string_lossy().to_string();
        }
    }
    format!("resources/models/{}", name)
}

fn letterbox_resize(img: &image::DynamicImage, target_w: u32, target_h: u32) -> (image::DynamicImage, i32, i32, i32, i32) {
    let (orig_w, orig_h) = img.dimensions();
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);

    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

    let mut canvas = image::RgbImage::from_pixel(target_w, target_h, image::Rgb([0, 0, 0]));
    let offset_x = ((target_w - new_w) / 2) as i32;
    let offset_y = ((target_h - new_h) / 2) as i32;

    image::imageops::overlay(&mut canvas, &resized.to_rgb8(), offset_x as i64, offset_y as i64);

    (image::DynamicImage::ImageRgb8(canvas), offset_x, offset_y, new_w as i32, new_h as i32)
}

fn main() {
    let img_path = "/Users/mac/Downloads/eAS4m39wfMgoYyA.thumb.1000_0.jpg";

    println!("=== SCRFD Raw Output Test (Detailed) ===");
    println!("Image: {}", img_path);

    let scrfd_path = find_model("scrfd_500m_bnkps.onnx");
    println!("Model: {}", scrfd_path);

    let img = image::open(img_path).expect("Failed to open image");
    let (orig_w, orig_h) = img.dimensions();
    println!("Original size: {}x{}", orig_w, orig_h);

    let (resized, pad_x, pad_y, new_w, new_h) = letterbox_resize(&img, INPUT_SIZE as u32, INPUT_SIZE as u32);
    let rgb = resized.to_rgb8();
    println!("Resized: {}x{}, pad=({}, {}), canvas=640x640", new_w, new_h, pad_x, pad_y);

    // BGR CHW input
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

    let mut session = Session::builder()
        .expect("Failed to create session")
        .commit_from_file(&scrfd_path)
        .expect("Failed to load model");
    let input = Tensor::from_array(([1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64], input_data))
        .map_err(|e| format!("Tensor error: {}", e)).unwrap();

    println!("Running inference...");
    let outputs = session.run(ort::inputs![input]).expect("Inference failed");

    println!("\n=== ONNX Output Shapes ===");
    for (i, output) in outputs.iter().enumerate() {
        let shape = output.1.shape();
        println!("Output {}: shape={:?}", i, shape);
    }

    println!("\n=== Detailed Decode Analysis ===");

    // Analyze scale 1 (stride=16) in detail - this is where detections typically happen
    let scale_idx = 1;
    let stride = STRIDES[scale_idx];

    let score_data = outputs[scale_idx].try_extract_tensor::<f32>().unwrap().1;
    let bbox_data = outputs[scale_idx + 3].try_extract_tensor::<f32>().unwrap().1;
    let kps_data = outputs[scale_idx + 6].try_extract_tensor::<f32>().unwrap().1;

    let num_anchors = score_data.len();
    let grid_w = INPUT_SIZE as usize / stride as usize;

    println!("\n--- Scale {} (stride={}) ---", scale_idx, stride);
    println!("num_anchors={}, grid_w={}", num_anchors, grid_w);
    println!("bbox_data len={}, kps_data len={}", bbox_data.len(), kps_data.len());

    // Check tensor layout by examining stride of indices
    println!("\n=== Tensor Layout Check ===");
    println!("If bbox layout is [N,4]: bbox[i*4] for i=0 gives indices 0,1,2,3");
    println!("If bbox layout is [4,N]: bbox[i] for i=0 gives all dx, bbox[N+i] gives all dy");

    // Find detections that would pass raw score threshold
    let mut pass_indices = Vec::new();
    for i in 0..num_anchors {
        let raw_score = score_data[i];
        if raw_score >= 0.3 {
            pass_indices.push(i);
        }
    }
    println!("\nTotal passing raw_score>=0.3: {}", pass_indices.len());
    println!("Sample indices: {:?}", pass_indices.iter().take(20).collect::<Vec<_>>());

    // Also find detections with high raw scores
    let mut high_score_indices: Vec<(usize, f32)> = (0..num_anchors)
        .map(|i| (i, score_data[i]))
        .filter(|(_, s)| *s > 0.1)
        .collect();
    high_score_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 20 by raw_score:");
    for (idx, score) in high_score_indices.iter().take(20) {
        println!("  i={} raw_score={:.4}", idx, score);
    }

    // Decode first few detections and compare layouts
    println!("\n=== Decode Comparison (i*4 vs N+i layout) ===");

    // Check specific indices that user mentioned might be problematic
    let check_indices: Vec<usize> = high_score_indices.iter().take(10).map(|(i, _)| *i).collect();

    for det_idx in &check_indices {
        let i = *det_idx;
        let row = i / grid_w;
        let col = i % grid_w;
        let anchor_cx = (col as f32 + 0.5) * stride;
        let anchor_cy = (row as f32 + 0.5) * stride;

        println!("\n--- Detection i={} ---", i);
        println!("anchor=({:.1}, {:.1})", anchor_cx, anchor_cy);

        // BBox Layout A: [N, 4] - current assumption
        let bbox_idx_a = i * 4;
        let dx_a = bbox_data[bbox_idx_a];
        let dy_a = bbox_data[bbox_idx_a + 1];
        let w_a = bbox_data[bbox_idx_a + 2];
        let h_a = bbox_data[bbox_idx_a + 3];

        // BBox Layout B: [4, N] - like photofinder-ai
        let dx_b = bbox_data[i];
        let dy_b = bbox_data[num_anchors + i];
        let w_b = bbox_data[2 * num_anchors + i];
        let h_b = bbox_data[3 * num_anchors + i];

        println!("BBox Layout A [N,4] (i*4): dx={:.3}, dy={:.3}, w={:.3}, h={:.3}", dx_a, dy_a, w_a, h_a);
        println!("BBox Layout B [4,N] (N+i):  dx={:.3}, dy={:.3}, w={:.3}, h={:.3}", dx_b, dy_b, w_b, h_b);

        // Decode using Formula A: (dx - 0.5) * stride * 2
        let cx_a = anchor_cx + (dx_a - 0.5) * stride * 2.0;
        let cy_a = anchor_cy + (dy_a - 0.5) * stride * 2.0;
        let bw_a = w_a * stride;
        let bh_a = h_a * stride;
        let x1_a = (cx_a - bw_a / 2.0).max(0.0).min(INPUT_SIZE as f32);
        let y1_a = (cy_a - bh_a / 2.0).max(0.0).min(INPUT_SIZE as f32);
        let x2_a = (cx_a + bw_a / 2.0).max(0.0).min(INPUT_SIZE as f32);
        let y2_a = (cy_a + bh_a / 2.0).max(0.0).min(INPUT_SIZE as f32);

        // Decode using Formula B: (dx - 0.5) * stride * 2 (same formula, different raw values)
        let cx_b = anchor_cx + (dx_b - 0.5) * stride * 2.0;
        let cy_b = anchor_cy + (dy_b - 0.5) * stride * 2.0;
        let bw_b = w_b * stride;
        let bh_b = h_b * stride;
        let x1_b = (cx_b - bw_b / 2.0).max(0.0).min(INPUT_SIZE as f32);
        let y1_b = (cy_b - bh_b / 2.0).max(0.0).min(INPUT_SIZE as f32);
        let x2_b = (cx_b + bw_b / 2.0).max(0.0).min(INPUT_SIZE as f32);
        let y2_b = (cy_b + bh_b / 2.0).max(0.0).min(INPUT_SIZE as f32);

        println!("Decoded A: face=({:.0},{:.0},{:.0},{:.0}) size=({:.0}x{:.0})", x1_a, y1_a, x2_a, y2_a, bw_a, bh_a);
        println!("Decoded B: face=({:.0},{:.0},{:.0},{:.0}) size=({:.0}x{:.0})", x1_b, y1_b, x2_b, y2_b, bw_b, bh_b);

        // KPS
        // KPS Layout A: [N, 10] - current assumption
        // Detection i's 10 values are at indices i*10, i*10+1, ..., i*10+9
        let kps_idx_a = i * 10;
        let kps_raw_a = &kps_data[kps_idx_a..kps_idx_a + 10];

        // KPS Layout B: [10, N] - transposed
        // Row j contains all detections' coordinate j
        // Detection i's coordinate j is at kps_data[j * N + i]
        let kps_raw_b: Vec<f32> = (0..10).map(|j| kps_data[j * num_anchors + i]).collect();

        println!("KPS Layout A [N,10] (i*10): raw=[{:.3},{:.3},{:.3},{:.3},{:.3}]", kps_raw_a[0], kps_raw_a[1], kps_raw_a[2], kps_raw_a[3], kps_raw_a[4]);
        println!("KPS Layout B [10,N] (j*N+i): raw=[{:.3},{:.3},{:.3},{:.3},{:.3}]", kps_raw_b[0], kps_raw_b[1], kps_raw_b[2], kps_raw_b[3], kps_raw_b[4]);

        // Decode KPS using Formula A: raw * stride
        let le_x_a = anchor_cx + kps_raw_a[0] * stride;
        let le_y_a = anchor_cy + kps_raw_a[1] * stride;
        let re_x_a = anchor_cx + kps_raw_a[2] * stride;
        let re_y_a = anchor_cy + kps_raw_a[3] * stride;
        let eye_dist_a = ((re_x_a - le_x_a).powi(2) + (re_y_a - le_y_a).powi(2)).sqrt();

        // Decode KPS using Formula B: raw * stride * 2
        let le_x_b = anchor_cx + kps_raw_a[0] * stride * 2.0;
        let le_y_b = anchor_cy + kps_raw_a[1] * stride * 2.0;
        let re_x_b = anchor_cx + kps_raw_a[2] * stride * 2.0;
        let re_y_b = anchor_cy + kps_raw_a[3] * stride * 2.0;
        let eye_dist_b = ((re_x_b - le_x_b).powi(2) + (re_y_b - le_y_b).powi(2)).sqrt();

        println!("KPS Decode A (raw*stride): le=({:.1},{:.1}) re=({:.1},{:.1}) eye_dist={:.1}", le_x_a, le_y_a, re_x_a, re_y_a, eye_dist_a);
        println!("KPS Decode B (raw*stride*2): le=({:.1},{:.1}) re=({:.1},{:.1}) eye_dist={:.1}", le_x_b, le_y_b, re_x_b, re_y_b, eye_dist_b);
    }

    println!("\n=== Test Complete ===");
}