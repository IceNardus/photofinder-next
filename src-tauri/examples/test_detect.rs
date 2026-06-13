use photofinder_next_lib::ai::face::detector::FaceDetector;

fn main() {
    let models_dir = std::path::PathBuf::from("/Users/mac/ai-project/photofinder-ai/src-tauri/resources/models");
    let scrfd_path = models_dir.join("scrfd_500m_bnkps.onnx");
    
    println!("Loading SCRFD from: {:?}", scrfd_path);
    let detector = FaceDetector::new(&scrfd_path.to_string_lossy()).expect("Failed to create detector");
    
    let img_path = "/Users/mac/Downloads/asian-beautiful-woman-with-brown-long-hair-portrait-white-tshirt-jean-jacket-costume-liftstyle-concept.jpg";
    println!("Detecting faces in: {}", img_path);
    println!("Image info: {:?}", image::open(img_path).unwrap().dimensions());
    
    let result = detector.detect(img_path);
    match result {
        Ok(faces) => {
            println!("Detected {} faces", faces.len());
            for (i, face) in faces.iter().enumerate() {
                println!("Face {}: bbox={:?}, score={:.4}", i, face.bbox, face.score);
            }
        }
        Err(e) => println!("Detection error: {}", e),
    }
}
