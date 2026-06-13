//! Test face pipeline using library components
use std::path::Path;
use std::sync::Mutex;
use ort::session::Session;
use ort::value::Tensor;
use image::{GenericImageView, imageops::FilterType};

const SCRFD_INPUT_SIZE: i32 = 640;
const ARCFACE_INPUT_SIZE: u32 = 112;
const MIN_CONFIDENCE: f32 = 0.5;
const MIN_FACE_SIZE: f32 = 10.0;
const NMS_IOU_THRESHOLD: f32 = 0.4;
const STRIDES: [f32; 3] = [8.0, 16.0, 32.0];

// Re-export from library
use photofinder_next_lib::ai::face::{
    detector::{FaceDetector, DetectedFace, FiveKeypoints},
    align::FaceAligner,
    arcface::ArcFace,
};

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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

fn main() {
    println!("=== Face Pipeline Test (Using Library Components) ===\n");

    let scrfd_path = find_model("scrfd_500m_bnkps.onnx");
    let arcface_path = find_model("w600k_r50.onnx");

    println!("SCRFD: {}", scrfd_path);
    println!("ArcFace: {}", arcface_path);

    let detector = FaceDetector::new(&scrfd_path).expect("Failed to create detector");
    let aligner = FaceAligner::new();
    let arcface = ArcFace::new(&arcface_path).expect("Failed to create arcface");

    let img1_path = "/Users/mac/Downloads/jJSdALjbCewl05W.thumb.1000_0.jpg";
    let img2_path = "/Users/mac/Downloads/73S2Y9lgheaynz0.thumb.1000_0.jpg";

    // Process image 1
    println!("\n--- Processing {} ---", img1_path);
    let faces1 = detector.detect(img1_path).expect("Detection failed");
    println!("Detected {} faces", faces1.len());

    let mut embedding1: Option<Vec<f32>> = None;
    if !faces1.is_empty() {
        match aligner.align(img1_path, &faces1[0].keypoints) {
            Ok(aligned) => {
                aligned.save("/tmp/lib_aligned1.jpg").ok();
                println!("Saved aligned face to /tmp/lib_aligned1.jpg");
                match arcface.extract(&aligned) {
                    Ok(emb) => {
                        println!("Embedding dim: {}", emb.len());
                        embedding1 = Some(emb);
                    }
                    Err(e) => println!("ArcFace error: {}", e),
                }
            }
            Err(e) => println!("Alignment error: {}", e),
        }
    }

    // Process image 2
    println!("\n--- Processing {} ---", img2_path);
    let faces2 = detector.detect(img2_path).expect("Detection failed");
    println!("Detected {} faces", faces2.len());

    let mut embedding2: Option<Vec<f32>> = None;
    if !faces2.is_empty() {
        match aligner.align(img2_path, &faces2[0].keypoints) {
            Ok(aligned) => {
                aligned.save("/tmp/lib_aligned2.jpg").ok();
                println!("Saved aligned face to /tmp/lib_aligned2.jpg");
                match arcface.extract(&aligned) {
                    Ok(emb) => {
                        println!("Embedding dim: {}", emb.len());
                        embedding2 = Some(emb);
                    }
                    Err(e) => println!("ArcFace error: {}", e),
                }
            }
            Err(e) => println!("Alignment error: {}", e),
        }
    }

    if let (Some(e1), Some(e2)) = (embedding1, embedding2) {
        let sim = cosine_similarity(&e1, &e2);
        println!("\n=== Cosine Similarity ===");
        println!("Similarity: {:.4}", sim);
        println!("Different person < 0.4: {}", if sim < 0.4 { "PASS" } else { "FAIL" });
    }

    println!("\n=== Test Complete ===");
}