use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tauri::State;
use anyhow::Result;
use tracing::{info, warn};
use rusqlite;
use image::DynamicImage;

use crate::core::database::Database;
use crate::core::database::images::ImagesTable;
use crate::core::database::tasks::TasksTable;
use crate::core::statistics::StatisticsCollector;
use crate::core::thumbnail::ThumbnailStore;
use crate::core::scanner::Scanner;
use crate::search::person_search::{PersonSearch, PersonSearchResult};
use crate::search::object_search::{ObjectSearch, ObjectSearchConfig, ObjectSearchResult as ObjSearchResult};
use crate::category_search::{CategorySearch, CategorySearchConfig, SearchResult as CategorySearchResult};

pub struct AppState {
    pub db: Arc<Database>,
    pub stats: StatisticsCollector,
    pub thumbnail_store: Arc<ThumbnailStore>,
    pub scanner: Arc<Scanner>,
    pub person_search: Arc<RwLock<PersonSearch>>,
    pub object_search: Arc<RwLock<Option<Arc<ObjectSearch>>>>,
    pub category_search: Arc<RwLock<Option<CategorySearch>>>,
    pub is_scanning: Arc<RwLock<bool>>,
    pub is_processing: Arc<RwLock<bool>>,  // Track background processing
    pub last_scan_stats: Arc<RwLock<ScanStats>>,
    pub processing_stats: Arc<RwLock<ProcessingStats>>,
}

// Safety: AppState contains only Send+Sync types (Arc<Mutex<T>>, Arc<RwLock<T>>, etc.)
// All ONNX sessions are wrapped in Mutex and use unsafe impl to claim Send+Sync
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

#[derive(Debug, Clone, Default)]
pub struct ScanStats {
    pub total_images: usize,
    pub processed_images: usize,
    pub pending_tasks: usize,
    pub current_file: String,
    pub current_faces: usize,
    pub current_patches: usize,
    pub current_objects: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanStatus {
    pub is_scanning: bool,
    pub total_images: usize,
    pub processed_images: usize,
    pub pending_tasks: usize,
    pub current_file: String,
    pub current_faces: usize,
    pub current_patches: usize,
    pub current_objects: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    pub current_image: String,
    pub current_faces: usize,
    pub current_patches: usize,
    pub current_objects: usize,
    pub log_message: String,
    pub last_completion_message: String, // Track last completion for frontend detection
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessingStatus {
    pub current_image: String,
    pub current_faces: usize,
    pub current_patches: usize,
    pub current_objects: usize,
    pub log_message: String,
    pub last_completion_message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Statistics {
    pub image_count: i64,
    pub face_count: i64,
    pub patch_count: i64,
    pub object_count: i64,
    pub pending_task_count: i64,
    pub index_size_bytes: u64,
    pub vector_store_size_bytes: u64,
    pub database_size_bytes: u64,
    pub thumbnail_count: usize,
    pub thumbnail_size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub image_id: i64,
    pub face_id: i64,
    pub thumbnail_path: String,
    pub similarity: f32,
    pub bbox: Option<[f32; 4]>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThumbnailProgress {
    pub total: usize,
    pub generated: usize,
    pub failed: usize,
    pub speed: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanResult {
    pub total_found: usize,
    pub new_images: usize,
    pub skipped: usize,
}

#[tauri::command]
pub async fn scan_folder(folder_path: String, state: State<'_, AppState>) -> Result<ScanResult, String> {
    info!("Starting scan for folder: {}", folder_path);

    *state.is_scanning.write().await = true;
    {
        let mut scan_stats = state.last_scan_stats.write().await;
        scan_stats.current_file = folder_path.clone();
        scan_stats.total_images = 0;
        scan_stats.processed_images = 0;
    }

    let path = std::path::PathBuf::from(&folder_path);
    let progress_stats = Arc::clone(&state.last_scan_stats);
    let result = state.scanner.scan_folder_with_progress(&path, Some(progress_stats)).await;

    *state.is_scanning.write().await = false;

    match result {
        Ok(r) => {
            // Update scan_stats with final counts after folder scan completes
            {
                let mut scan_stats = state.last_scan_stats.write().await;
                scan_stats.total_images = r.total_found;
                scan_stats.processed_images = r.total_found; // All found images are "added" after scan
            }
            Ok(ScanResult {
                total_found: r.total_found,
                new_images: r.new_images,
                skipped: r.skipped,
            })
        }
        Err(e) => {
            tracing::error!("Scan failed: {}", e);
            Err(e.to_string())
        }
    }
}

#[tauri::command]
pub async fn stop_scan(_state: State<'_, AppState>) -> Result<(), String> {
    info!("Stopping scan");
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct ClearDatabaseResult {
    pub success: bool,
    pub cleared_images: usize,
    pub cleared_faces: usize,
    pub cleared_thumbnails: usize,
    pub cleared_vectors: bool,
    pub cleared_category_index: bool,
    pub errors: Vec<String>,
}

#[tauri::command]
pub async fn clear_database(state: State<'_, AppState>) -> Result<ClearDatabaseResult, String> {
    info!("clear_database command called");

    let mut result = ClearDatabaseResult {
        success: true,
        cleared_images: 0,
        cleared_faces: 0,
        cleared_thumbnails: 0,
        cleared_vectors: false,
        cleared_category_index: false,
        errors: vec![],
    };

    // Get data directory
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("PhotoFinderNext");

    // Clear photofinder.db tables
    match state.db.clear_all() {
        Ok(()) => {
            info!("[CLEAR] Database tables cleared");
        }
        Err(e) => {
            let msg = format!("Database clear failed: {}", e);
            eprintln!("[CLEAR] {}", msg);
            result.errors.push(msg);
            result.success = false;
        }
    }

    // Clear thumbnails
    match state.thumbnail_store.clear_all() {
        Ok(()) => {
            result.cleared_thumbnails = 1;
            info!("[CLEAR] Thumbnails cleared");
        }
        Err(e) => {
            let msg = format!("Thumbnail clear failed: {}", e);
            eprintln!("[CLEAR] {}", msg);
            result.errors.push(msg);
        }
    }

    // Clear category_index.db
    let category_db_path = data_dir.join("category_index.db");
    if category_db_path.exists() {
        match rusqlite::Connection::open(&category_db_path) {
            Ok(conn) => {
                let _ = conn.execute("DELETE FROM category_embeddings", []);
                let _ = conn.execute("DELETE FROM category_objects", []);
                result.cleared_category_index = true;
                info!("[CLEAR] category_index.db cleared");
            }
            Err(e) => {
                let msg = format!("category_index.db clear failed: {}", e);
                eprintln!("[CLEAR] {}", msg);
                result.errors.push(msg);
            }
        }
    }

    // Clear all files in vectors directory
    let vectors_dir = data_dir.join("vectors");
    if vectors_dir.exists() {
        match std::fs::read_dir(&vectors_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                        if let Err(e) = std::fs::remove_file(entry.path()) {
                            let msg = format!("Failed to delete {:?}: {}", entry.path(), e);
                            eprintln!("[CLEAR] {}", msg);
                            result.errors.push(msg);
                        }
                    }
                }
                result.cleared_vectors = true;
                info!("[CLEAR] Vectors directory cleared");
            }
            Err(e) => {
                let msg = format!("Failed to read vectors dir: {}", e);
                eprintln!("[CLEAR] {}", msg);
                result.errors.push(msg);
            }
        }
    }

    // Clear person cluster index
    let person_index_path = vectors_dir.join("person_cluster.bin");
    if person_index_path.exists() {
        if let Err(e) = std::fs::remove_file(&person_index_path) {
            eprintln!("[CLEAR] Failed to delete person_cluster.bin: {}", e);
        } else {
            info!("[CLEAR] person_cluster.bin deleted");
        }
    }

    // Clear debug directory
    let debug_dir = data_dir.join("debug");
    if debug_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&debug_dir) {
            for entry in entries.flatten() {
                if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    // Reset scan state
    *state.is_scanning.write().await = false;
    *state.is_processing.write().await = false;
    *state.last_scan_stats.write().await = ScanStats::default();
    *state.processing_stats.write().await = ProcessingStats::default();

    info!("clear_database completed. success={}, errors={}", result.success, result.errors.len());
    Ok(result)
}

#[tauri::command]
pub async fn rebuild_thumbnails(state: State<'_, AppState>) -> Result<ThumbnailProgress, String> {
    info!("Rebuilding all thumbnails");
    let start = Instant::now();

    let images = ImagesTable::get_all_for_thumbnails(&state.db.conn)
        .map_err(|e| e.to_string())?;

    let total = images.len();
    let mut generated = 0;
    let mut failed = 0;

    for image in &images {
        match state.thumbnail_store.generate(&image.path, &image.hash) {
            Ok(thumb_path) => {
                ImagesTable::update_thumbnail(
                    &state.db.conn,
                    &image.hash,
                    &thumb_path,
                    "generated"
                ).map_err(|e| e.to_string())?;
                generated += 1;
            }
            Err(e) => {
                tracing::warn!("Failed to generate thumbnail for {}: {}", image.path, e);
                ImagesTable::update_thumbnail(
                    &state.db.conn,
                    &image.hash,
                    "",
                    "failed"
                ).map_err(|e| e.to_string())?;
                failed += 1;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let speed = state.stats.thumbnail_generation_speed(elapsed, generated);

    info!("Thumbnail rebuild complete: {} generated, {} failed, {:.2} img/s", generated, failed, speed);

    Ok(ThumbnailProgress {
        total,
        generated,
        failed,
        speed,
    })
}

#[tauri::command]
pub async fn search_person(query_image: String, top_k: usize, state: State<'_, AppState>) -> Result<Vec<SearchResult>, String> {
    let searcher = state.person_search.read().await;
    match searcher.search(&query_image, top_k).await {
        Ok(results) => Ok(results.into_iter().map(|r| SearchResult {
            image_id: r.image_id,
            face_id: r.face_id,
            thumbnail_path: r.thumbnail_path,
            similarity: r.score,
            bbox: r.bbox,
        }).collect()),
        Err(e) => Err(e.to_string()),
    }
}

#[derive(Debug, Serialize)]
pub struct LibraryImage {
    pub id: i64,
    pub path: String,
    pub thumbnail_path: String,
}

#[tauri::command]
pub async fn get_library_images(state: State<'_, AppState>) -> Result<Vec<LibraryImage>, String> {
    let images = ImagesTable::get_all(&state.db.conn, 200, 0)
        .map_err(|e| e.to_string())?;

    Ok(images.into_iter().filter_map(|img| {
        let thumb = img.thumbnail_path?;
        Some(LibraryImage {
            id: img.id,
            path: img.path,
            thumbnail_path: thumb,
        })
    }).collect())
}

#[tauri::command]
pub async fn get_scan_status(state: State<'_, AppState>) -> Result<ScanStatus, String> {
    // is_scanning is true if folder scan is active OR there are pending tasks
    let folder_scanning = *state.is_scanning.read().await;
    let pending_tasks = TasksTable::pending_count(&state.db.conn).unwrap_or(0) as usize;
    let image_count = ImagesTable::count(&state.db.conn).unwrap_or(0) as usize;
    let scan_stats = state.last_scan_stats.read().await;
    let proc_stats = state.processing_stats.read().await;

    // Show scanning if folder scan is active OR there are pending tasks being processed
    let is_scanning = folder_scanning || pending_tasks > 0;

    // Use current_image from processing_stats if available (background processing)
    let current_file = if !proc_stats.current_image.is_empty() {
        std::path::Path::new(&proc_stats.current_image).file_name().and_then(|s| s.to_str())
            .unwrap_or(&proc_stats.current_image).to_string()
    } else {
        scan_stats.current_file.clone()
    };

    // Calculate processed_images: during folder scan use scan_stats,
    // during background processing calculate from pending tasks
    let processed_images = if folder_scanning {
        scan_stats.processed_images
    } else if pending_tasks > 0 {
        // Background processing: total DB images - pending = actually processed
        image_count.saturating_sub(pending_tasks)
    } else {
        // Idle: all images are processed
        image_count
    };

    Ok(ScanStatus {
        is_scanning,
        total_images: image_count,
        processed_images,
        pending_tasks,
        current_file,
        current_faces: proc_stats.current_faces,
        current_patches: proc_stats.current_patches,
        current_objects: proc_stats.current_objects,
    })
}

#[tauri::command]
pub async fn get_processing_status(state: State<'_, AppState>) -> Result<ProcessingStatus, String> {
    let stats = state.processing_stats.read().await;
    Ok(ProcessingStatus {
        current_image: stats.current_image.clone(),
        current_faces: stats.current_faces,
        current_patches: stats.current_patches,
        current_objects: stats.current_objects,
        log_message: stats.log_message.clone(),
        last_completion_message: stats.last_completion_message.clone(),
    })
}

#[tauri::command]
pub async fn get_statistics(state: State<'_, AppState>) -> Result<Statistics, String> {
    let image_count = ImagesTable::count(&state.db.conn).unwrap_or(0);
    let face_count = crate::core::database::faces::FacesTable::count(&state.db.conn).unwrap_or(0);
    let patch_count = crate::core::database::patches::PatchFeaturesTable::count(&state.db.conn).unwrap_or(0);
    let pending_task_count = TasksTable::pending_count(&state.db.conn).unwrap_or(0);

    // Get object count from category_index.db
    let object_count = {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let category_db_path = data_dir.join("PhotoFinderNext").join("category_index.db");
        if let Ok(conn) = rusqlite::Connection::open(&category_db_path) {
            conn.query_row("SELECT COUNT(*) FROM category_objects", [], |row| row.get(0))
                .unwrap_or(0)
        } else {
            0
        }
    };

    Ok(Statistics {
        image_count,
        face_count,
        patch_count,
        object_count,
        pending_task_count,
        index_size_bytes: 0,
        vector_store_size_bytes: 0,
        database_size_bytes: 0,
        thumbnail_count: state.thumbnail_store.count(),
        thumbnail_size_bytes: state.thumbnail_store.calculate_size(),
    })
}

#[derive(Debug, Serialize)]
pub struct FacePipelineResult {
    pub face_count: usize,
    pub quality: f32,
    pub embedding_dimension: usize,
}

#[tauri::command]
pub async fn test_face_pipeline(image_path: String) -> Result<FacePipelineResult, String> {
    use crate::ai::face::{FacePipeline, FaceDetector, FaceAligner, ArcFace};

    let find_model = |name: &str| -> String {
        crate::core::models::find_model_path(name).to_string_lossy().to_string()
    };

    let scrfd_path = find_model("scrfd_500m_bnkps.onnx");
    let arcface_path = find_model("w600k_r50.onnx");

    tracing::info!("Loading SCRFD from: {}", scrfd_path);
    tracing::info!("Loading ArcFace from: {}", arcface_path);

    let detector = FaceDetector::new(&scrfd_path).map_err(|e| e.to_string())?;
    let aligner = FaceAligner::new();
    let arcface = ArcFace::new(&arcface_path).map_err(|e| e.to_string())?;

    let debug_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("PhotoFinderNext")
        .join("debug");
    let pipeline = FacePipeline::with_debug(detector, aligner, arcface, &debug_dir);

    let features = pipeline.process_image(&image_path).map_err(|e| e.to_string())?;

    let face_count = features.len();
    let avg_quality = if face_count > 0 {
        features.iter().map(|f| f.quality).sum::<f32>() / face_count as f32
    } else {
        0.0
    };
    let embedding_dimension = pipeline.embedding_dim();

    tracing::info!("Face pipeline test complete: {} faces, quality={:.3}, dim={}",
        face_count, avg_quality, embedding_dimension);

    Ok(FacePipelineResult {
        face_count,
        quality: avg_quality,
        embedding_dimension,
    })
}

#[derive(Debug, Serialize)]
pub struct FaceSimilarityResult {
    pub similarity: f32,
    pub face1_count: usize,
    pub face2_count: usize,
    pub embedding1_dim: usize,
    pub embedding2_dim: usize,
}

#[tauri::command]
pub async fn test_face_similarity(image1_path: String, image2_path: String) -> Result<FaceSimilarityResult, String> {
    use crate::ai::face::{FacePipeline, FaceDetector, FaceAligner, ArcFace};

    let find_model = |name: &str| -> String {
        crate::core::models::find_model_path(name).to_string_lossy().to_string()
    };

    let scrfd_path = find_model("scrfd_500m_bnkps.onnx");
    let arcface_path = find_model("w600k_r50.onnx");

    let detector = FaceDetector::new(&scrfd_path).map_err(|e| e.to_string())?;
    let aligner = FaceAligner::new();
    let arcface = ArcFace::new(&arcface_path).map_err(|e| e.to_string())?;

    let debug_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("PhotoFinderNext")
        .join("debug");
    let pipeline = FacePipeline::with_debug(detector, aligner, arcface, &debug_dir);

    let features1 = pipeline.process_image(&image1_path).map_err(|e| e.to_string())?;
    let embedding1 = if features1.is_empty() {
        vec![0.0; 512]
    } else {
        features1[0].embedding.clone()
    };

    let features2 = pipeline.process_image(&image2_path).map_err(|e| e.to_string())?;
    let embedding2 = if features2.is_empty() {
        vec![0.0; 512]
    } else {
        features2[0].embedding.clone()
    };

    let dot: f32 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum();
    let norm1 = embedding1.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm2 = embedding2.iter().map(|v| v * v).sum::<f32>().sqrt();
    let similarity = if norm1 > 0.0 && norm2 > 0.0 { dot / (norm1 * norm2) } else { 0.0 };

    tracing::info!("Face similarity: {} vs {} = {:.4}", image1_path, image2_path, similarity);

    Ok(FaceSimilarityResult {
        similarity,
        face1_count: features1.len(),
        face2_count: features2.len(),
        embedding1_dim: embedding1.len(),
        embedding2_dim: embedding2.len(),
    })
}

#[derive(Debug, Serialize)]
pub struct AlignmentScaleResult {
    pub scale: f32,
    pub face_coverage: f32,
    pub avg_luminance: f32,
    pub luminance_range: f32,
    pub is_valid: bool,
    pub bbox: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct FaceAlignmentAnalysis {
    pub image_path: String,
    pub detected_faces: usize,
    pub scale_results: Vec<Vec<AlignmentScaleResult>>,  // [face][scale]
    pub optimal_scale: Option<f32>,
    pub optimal_face_index: usize,
}

#[tauri::command]
pub async fn analyze_face_alignment(image_path: String) -> Result<FaceAlignmentAnalysis, String> {
    use crate::ai::face::{FaceDetector, FaceAligner, FaceAlignmentConfig, scan_alignment_scales};

    let find_model = |name: &str| -> String {
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        candidates.push(std::path::PathBuf::from("resources/models").join(name));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                candidates.push(parent.join("resources/models").join(name));
            }
        }
        candidates.push(std::path::PathBuf::from("/Applications/PhotoFinder Next.app/Contents/Resources/models").join(name));
        candidates.push(std::path::PathBuf::from("/Users/mac/Library/Caches/PhotoFinder/models").join(name));
        candidates.push(std::path::PathBuf::from("/Users/mac/ai-project/photofinder-ai/src-tauri/resources/models").join(name));
        for candidate in &candidates {
            if candidate.exists() {
                return candidate.to_string_lossy().to_string();
            }
        }
        format!("resources/models/{}", name)
    };

    let scrfd_path = find_model("scrfd_500m_bnkps.onnx");
    let detector = FaceDetector::new(&scrfd_path).map_err(|e| e.to_string())?;

    let detected_faces = detector.detect(&image_path).map_err(|e| e.to_string())?;
    let face_count = detected_faces.len();

    let candidate_scales = [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4];
    let mut all_results: Vec<Vec<AlignmentScaleResult>> = Vec::new();
    let mut best_overall_score = 0.0f32;
    let mut best_overall_scale = None;
    let mut best_face_index = 0;

    for (i, face) in detected_faces.iter().enumerate() {
        let scale_results = scan_alignment_scales(&image_path, &face.keypoints, &candidate_scales);

        let face_results: Vec<AlignmentScaleResult> = scale_results
            .iter()
            .map(|(scale, metrics)| AlignmentScaleResult {
                scale: *scale,
                face_coverage: metrics.face_coverage,
                avg_luminance: metrics.avg_luminance,
                luminance_range: metrics.luminance_range,
                is_valid: metrics.is_valid,
                bbox: face.bbox.to_vec(),
            })
            .collect();

        // Find best scale for this face
        for (scale, metrics) in &scale_results {
            if !metrics.is_valid {
                continue;
            }
            let overall_score = metrics.face_coverage * 0.7 + (metrics.luminance_range / 255.0) * 0.3;
            if overall_score > best_overall_score {
                best_overall_score = overall_score;
                best_overall_scale = Some(*scale);
                best_face_index = i;
            }
        }

        all_results.push(face_results);
    }

    tracing::info!("Face alignment analysis: {} faces, optimal_scale={:?}", face_count, best_overall_scale);

    Ok(FaceAlignmentAnalysis {
        image_path,
        detected_faces: face_count,
        scale_results: all_results,
        optimal_scale: best_overall_scale,
        optimal_face_index: best_face_index,
    })
}

#[tauri::command]
pub async fn create_temp_dir(dir: String) -> Result<(), String> {
    let base = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data local dir".to_string())?;
    let full_path = base.join(&dir);
    std::fs::create_dir_all(&full_path).map_err(|e| e.to_string())?;
    info!("[TEMP] Created dir: {:?}", full_path);
    Ok(())
}

#[tauri::command]
pub async fn write_temp_file(path: String, data: Vec<u8>) -> Result<(), String> {
    let base = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data local dir".to_string())?;
    let full_path = if std::path::Path::new(&path).is_absolute() {
        std::path::PathBuf::from(&path)
    } else {
        base.join(&path)
    };
    info!("[TEMP] write_temp_file: {:?}", full_path);
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&full_path, &data).map_err(|e| e.to_string())?;
    info!("[TEMP] Wrote {} bytes to {:?}", data.len(), full_path);
    Ok(())
}

#[tauri::command]
pub async fn ensure_temp_dir(dir: String) -> Result<(), String> {
    let base = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data local dir".to_string())?;
    let full_path = base.join(&dir);
    std::fs::create_dir_all(&full_path).map_err(|e| e.to_string())?;
    info!("[TEMP] ensure_temp_dir: {:?}", full_path);
    Ok(())
}

#[tauri::command]
pub async fn write_query_image(data: Vec<u8>, mime_type: String) -> Result<String, String> {
    let base = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data local dir".to_string())?
        .join("PhotoFinderNext");
    std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;
    if data.len() < 4 {
        return Err("Data too short".to_string());
    }
    let ext = match mime_type.as_str() {
        "image/jpeg" | "image/jpg" => "jpg",
        "image/png" => "png",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/bmp" => "bmp",
        _ => "jpg",
    };
    let query_path = base.join(format!("query_image.{}", ext));
    eprintln!("[QUERY] write_query_image: Writing {} bytes to {:?}", data.len(), query_path);
    std::fs::write(&query_path, &data).map_err(|e| e.to_string())?;
    eprintln!("[QUERY] write_query_image: Write complete, file now exists={}", query_path.exists());
    info!("[QUERY] Wrote {} bytes to {:?}", data.len(), query_path);
    let header = &data[..data.len().min(16)];
    info!("[QUERY] First bytes: {:?}", header);
    Ok(query_path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn write_cropped_image(data: Vec<u8>, _mime_type: String, x: u32, y: u32, w: u32, h: u32) -> Result<String, String> {
    use image::io::Reader as ImageReader;
    use image::ImageBuffer;

    // Use tracing (writes to log file) AND eprintln (stderr)
    info!("[CROP] write_cropped_image ENTRY: x={}, y={}, w={}, h={}, data_len={}", x, y, w, h, data.len());
    eprintln!("[CROP] write_cropped_image ENTRY: x={}, y={}, w={}, h={}, data_len={}", x, y, w, h, data.len());

    // Validate dimensions
    if w == 0 || h == 0 {
        let msg = format!("Invalid crop dimensions: {}x{}", w, h);
        info!("[CROP] ERROR: {}", msg);
        eprintln!("[CROP] ERROR: {}", msg);
        return Err(msg);
    }

    let base = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data local dir".to_string())?
        .join("PhotoFinderNext");
    std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;

    // Decode the image
    let img = ImageReader::new(std::io::Cursor::new(&data))
        .with_guessed_format()
        .map_err(|e| {
            let msg = format!("Format error: {}", e);
            info!("[CROP] {}", msg);
            eprintln!("[CROP] {}", msg);
            msg
        })?
        .decode()
        .map_err(|e| {
            let msg = format!("Decode error: {}", e);
            info!("[CROP] {}", msg);
            eprintln!("[CROP] {}", msg);
            format!("Failed to decode image: {}", e)
        })?;

    info!("[CROP] Image decoded: {}x{}", img.width(), img.height());
    eprintln!("[CROP] Image decoded: {}x{}", img.width(), img.height());

    // Check if crop region is valid
    if x + w > img.width() || y + h > img.height() {
        let msg = format!("Crop region {}x{} at ({},{}) exceeds image bounds {}x{}",
                          w, h, x, y, img.width(), img.height());
        info!("[CROP] ERROR: {}", msg);
        eprintln!("[CROP] ERROR: {}", msg);
        return Err(msg);
    }

    // Additional safety checks to prevent panics
    if w == 0 || h == 0 || x >= img.width() || y >= img.height() {
        let msg = format!("Invalid crop region: {}x{} at ({},{}) for image {}x{}",
                          w, h, x, y, img.width(), img.height());
        info!("[CROP] ERROR: {}", msg);
        eprintln!("[CROP] ERROR: {}", msg);
        return Err(msg);
    }

    // Crop the image and convert to RGB8 for reliable PNG saving
    let cropped = img.crop_imm(x, y, w, h);
    let rgb = cropped.to_rgb8();

    // Save cropped image - use same naming pattern as query_image for consistency
    let crop_path = base.join(format!("query_crop.png"));
    rgb.save(&crop_path).map_err(|e| {
        let msg = format!("Save error: {}", e);
        info!("[CROP] {}", msg);
        eprintln!("[CROP] {}", msg);
        format!("Failed to save crop: {}", e)
    })?;

    info!("[CROP] SUCCESS: {}x{} at ({},{}) saved to {:?}", w, h, x, y, crop_path);
    eprintln!("[CROP] SUCCESS: {}x{} at ({},{}) saved to {:?}", w, h, x, y, crop_path);
    Ok(crop_path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn get_image_thumbnail(image_path: String) -> Result<String, String> {
    use std::fs;
    use std::path::Path;
    eprintln!("[THUMB] get_image_thumbnail called with path: {}", image_path);
    let path = Path::new(&image_path);
    if !path.exists() {
        eprintln!("[THUMB] File not found: {}", image_path);
        return Err(format!("File not found: {}", image_path));
    }
    let data = fs::read(path).map_err(|e| {
        eprintln!("[THUMB] Failed to read file: {}", e);
        format!("Failed to read file: {}", e)
    })?;
    eprintln!("[THUMB] Read {} bytes from {}", data.len(), image_path);
    let mime = if image_path.ends_with(".png") {
        "image/png"
    } else if image_path.ends_with(".webp") {
        "image/webp"
    } else if image_path.ends_with(".gif") {
        "image/gif"
    } else {
        "image/jpeg"
    };
    let base64 = base64_encode(&data);
    eprintln!("[THUMB] Encoded {} bytes to base64 (len={})", data.len(), base64.len());
    Ok(format!("data:{};base64,{}", mime, base64))
}

fn base64_encode(data: &[u8]) -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    STANDARD.encode(data)
}

#[tauri::command]
pub async fn rebuild_face_index(state: State<'_, AppState>) -> Result<String, String> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    info!("[REBUILD] Starting face index rebuild...");

    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data dir".to_string())?
        .join("PhotoFinderNext");
    let bin_path = data_dir.join("vectors").join("face_vectors.bin");
    let meta_path = data_dir.join("vectors").join("face_vectors.meta");

    let faces = {
        let conn = state.db.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, vector_offset, vector_length FROM faces WHERE vector_offset > 0 AND vector_length > 0 ORDER BY id"
        ).map_err(|e| e.to_string())?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i32>(2)?))
        }).map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?
    };

    info!("[REBUILD] Found {} faces in database", faces.len());
    if faces.is_empty() {
        return Ok("No faces with vectors in database".to_string());
    }

    let all_vectors = if bin_path.exists() {
        let metadata = std::fs::metadata(&bin_path).map_err(|e| e.to_string())?;
        if metadata.len() > 92 {
            let mut old_file = std::fs::File::open(&bin_path).map_err(|e| e.to_string())?;
            let mut vectors = Vec::new();
            for (face_id, offset, length) in &faces {
                if *offset as u64 + *length as u64 > metadata.len() {
                    info!("[REBUILD] Skipping face {} - offset {} out of bounds", face_id, offset);
                    continue;
                }
                let mut buffer = vec![0u8; *length as usize];
                old_file.seek(SeekFrom::Start(*offset as u64)).map_err(|e| e.to_string())?;
                if old_file.read_exact(&mut buffer).is_err() {
                    info!("[REBUILD] Skipping face {} - read failed", face_id);
                    continue;
                }
                let embedding: Vec<f32> = buffer.chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                vectors.push((*face_id, embedding));
            }
            drop(old_file);
            vectors
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    info!("[REBUILD] Read {} vectors from old bin", all_vectors.len());

    if bin_path.exists() {
        std::fs::remove_file(&bin_path).map_err(|e| e.to_string())?;
    }
    if meta_path.exists() {
        std::fs::remove_file(&meta_path).map_err(|e| e.to_string())?;
    }

    let mut new_file = std::fs::File::create(&bin_path).map_err(|e| e.to_string())?;
    new_file.write_all(b"FVECTBIN").map_err(|e| e.to_string())?;
    new_file.write_all(&1u32.to_le_bytes()).map_err(|e| e.to_string())?;
    new_file.write_all(&512u32.to_le_bytes()).map_err(|e| e.to_string())?;
    new_file.write_all(&(all_vectors.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    let model_name = "ArcFace_w600k_r50".to_string();
    let mut padded = vec![0u8; 64];
    padded[..model_name.len()].copy_from_slice(model_name.as_bytes());
    new_file.write_all(&padded).map_err(|e| e.to_string())?;

    let mut new_meta_entries = Vec::new();
    let header_size = 92;
    let mut offset = header_size as u64;

    for (face_id, embedding) in &all_vectors {
        let data: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        new_file.write_all(&data).map_err(|e| e.to_string())?;
        new_meta_entries.push(serde_json::json!({
            "id": face_id,
            "offset": offset,
            "length": data.len() as u32
        }));
        offset += data.len() as u64;
    }

    let meta_json = serde_json::to_string(&new_meta_entries).map_err(|e| e.to_string())?;
    std::fs::write(&meta_path, meta_json).map_err(|e| e.to_string())?;

    info!("[REBUILD] Wrote {} vectors to new bin", all_vectors.len());
    Ok(format!("Rebuilt {} vectors with correct IDs", all_vectors.len()))
}

#[tauri::command]
pub async fn debug_faces(state: State<'_, AppState>) -> Result<String, String> {
    let mut output = String::new();

    let faces = {
        let conn = state.db.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, vector_offset, vector_length FROM faces ORDER BY id"
        ).map_err(|e| e.to_string())?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i32>(3)?,
            ))
        }).map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?
    };

    output.push_str(&format!("Faces in DB (count={}):\n", faces.len()));
    for (id, image_id, offset, length) in &faces {
        output.push_str(&format!("  face_id={}, image_id={}, offset={}, length={}\n", id, image_id, offset, length));
    }

    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data dir".to_string())?
        .join("PhotoFinderNext");
    let meta_path = data_dir.join("vectors").join("face_vectors.meta");

    if meta_path.exists() {
        let meta_content = std::fs::read_to_string(&meta_path).map_err(|e| e.to_string())?;
        output.push_str(&format!("\nMeta file content:\n{}\n", meta_content));
    } else {
        output.push_str("\nMeta file does NOT exist\n");
    }

    let images = {
        let conn = state.db.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path FROM images ORDER BY id LIMIT 20"
        ).map_err(|e| e.to_string())?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        }).map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?
    };

    output.push_str(&format!("\nImages in DB (count={}):\n", images.len()));
    for (id, path) in &images {
        let short_path = if path.len() > 60 { &path[path.len()-60..] } else { path };
        output.push_str(&format!("  id={}, path=...{}\n", id, short_path));
    }

    Ok(output)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ObjectSearchResultResponse {
    pub image_id: i64,
    pub image_path: String,
    pub image_name: String,        // 图片文件名
    pub thumbnail_path: String,
    pub confidence: f32,           // 融合分数
    pub similarity: f32,          // 语义相似度 (HNSW score)
    pub embedding_score: f32,     // MobileCLIP embedding score
    pub inlier_ratio: f32,       // LightGlue 内点比例
    pub inlier_count: usize,     // 内点数量
    pub match_count: usize,       // 匹配点数量
    pub bbox_overlap: f32,        // BBox重叠率
}

#[tauri::command]
pub async fn init_object_search(state: State<'_, AppState>) -> Result<String, String> {
    use crate::search::object_search::ObjectSearch;

    // Find models directory using cross-platform resolver
    let models_dir = match crate::core::models::find_models_dir() {
        Some(dir) => dir,
        None => return Err("Models directory not found. Please ensure models are installed.".to_string()),
    };

    if !models_dir.exists() {
        return Err(format!("Models directory not found: {}", models_dir.display()));
    }

    let mobileclip_path = models_dir.join("mobileclip_s2.onnx").to_string_lossy().to_string();
    let superpoint_path = models_dir.join("superpoint.onnx").to_string_lossy().to_string();
    let lightglue_path = models_dir.join("lightglue.onnx").to_string_lossy().to_string();

    if !std::path::Path::new(&mobileclip_path).exists() {
        return Err(format!("MobileCLIP model not found: {}", mobileclip_path));
    }
    if !std::path::Path::new(&superpoint_path).exists() {
        return Err(format!("SuperPoint model not found: {}", superpoint_path));
    }
    if !std::path::Path::new(&lightglue_path).exists() {
        return Err(format!("LightGlue model not found: {}", lightglue_path));
    }

    let config = ObjectSearchConfig::default();
    let object_search = ObjectSearch::new(
        &mobileclip_path,
        &superpoint_path,
        &lightglue_path,
        config,
    ).map_err(|e| e.to_string())?;

    // Initialize the category index for HNSW search
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("PhotoFinderNext");
    let category_db_path = data_dir.join("category_index.db");
    if let Err(e) = object_search.init_index(&category_db_path) {
        warn!("[OBJECT_SEARCH] Failed to init category index: {}", e);
        // Continue anyway - index might be empty
    }

    *state.object_search.write().await = Some(Arc::new(object_search));
    info!("[OBJECT_SEARCH] Initialized successfully");
    Ok("ObjectSearch initialized".to_string())
}

async fn get_or_init_object_search(state: &State<'_, AppState>) -> Result<Arc<ObjectSearch>, String> {
    // Fast path: already initialized
    {
        let object_search = state.object_search.read().await;
        if let Some(ref searcher) = *object_search {
            return Ok(Arc::clone(searcher));
        }
    }

    // Slow path: need to initialize
    info!("[OBJECT_SEARCH] Auto-initializing ObjectSearch...");
    use crate::search::object_search::ObjectSearch;

    let models_dir = match crate::core::models::find_models_dir() {
        Some(dir) => dir,
        None => return Err("Models directory not found. Please ensure models are installed.".to_string()),
    };

    if !models_dir.exists() {
        return Err(format!("Models directory not found: {}", models_dir.display()));
    }

    let mobileclip_path = models_dir.join("mobileclip_s2.onnx").to_string_lossy().to_string();
    let superpoint_path = models_dir.join("superpoint.onnx").to_string_lossy().to_string();
    let lightglue_path = models_dir.join("lightglue.onnx").to_string_lossy().to_string();

    let config = ObjectSearchConfig::default();
    let object_search = ObjectSearch::new(
        &mobileclip_path,
        &superpoint_path,
        &lightglue_path,
        config,
    ).map_err(|e| e.to_string())?;

    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("PhotoFinderNext");
    let index_dir = data_dir.join("vectors");  // Must match ProcessingService path!
    eprintln!("[OBJECT_SEARCH] Using index_dir: {}", index_dir.display());
    if let Err(e) = object_search.init_index(&index_dir) {
        warn!("[OBJECT_SEARCH] Failed to init category index: {}", e);
    }

    let searcher = Arc::new(object_search);
    *state.object_search.write().await = Some(Arc::clone(&searcher));
    info!("[OBJECT_SEARCH] Auto-initialized successfully");

    Ok(searcher)
}

#[tauri::command]
pub async fn search_objects(query_image: String, top_k: usize, state: State<'_, AppState>) -> Result<Vec<ObjectSearchResultResponse>, String> {
    let searcher = get_or_init_object_search(&state).await?;

    let results = searcher.search(&query_image, top_k).map_err(|e| e.to_string())?;

    // Get thumbnail paths from database
    let image_ids: Vec<i64> = results.iter().map(|r| r.image_id).collect();
    let (thumbnail_map, name_map) = {
        let conn = state.db.conn.lock().unwrap();
        let mut thumb_map = std::collections::HashMap::new();
        let mut name_map = std::collections::HashMap::new();
        for &img_id in &image_ids {
            let mut stmt = conn.prepare("SELECT thumbnail_path, path FROM images WHERE id = ?")
                .map_err(|e| e.to_string())?;
            if let Ok((thumb, path)) = stmt.query_row([img_id], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))) {
                thumb_map.insert(img_id, thumb);
                // Extract filename from path
                let name = std::path::Path::new(&path)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                name_map.insert(img_id, name);
            }
        }
        (thumb_map, name_map)
    };

    Ok(results.into_iter().map(|r| {
        // Use thumbnail_path if available, otherwise fall back to image_path (like person search)
        let thumb_or_path = thumbnail_map.get(&r.image_id)
            .cloned()
            .unwrap_or_else(|| r.image_path.clone());
        ObjectSearchResultResponse {
            image_id: r.image_id,
            image_path: r.image_path.clone(),
            image_name: name_map.get(&r.image_id).cloned().unwrap_or_default(),
            thumbnail_path: thumb_or_path,
            confidence: r.confidence,
            similarity: r.embedding_score,  // alias for compatibility
            embedding_score: r.embedding_score,
            inlier_ratio: r.inlier_ratio,
            inlier_count: r.inlier_count,
            match_count: r.match_count,
            bbox_overlap: r.bbox_overlap,
        }
    }).collect())
}

#[tauri::command]
pub async fn index_images_for_object(state: State<'_, AppState>) -> Result<String, String> {
    let object_search = state.object_search.read().await;
    let searcher = match object_search.as_ref() {
        Some(s) => s,
        None => return Err("ObjectSearch not initialized. Please click init_object_search first.".to_string()),
    };

    // Get all images from database
    let images = {
        let conn = state.db.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, path FROM images ORDER BY id")
            .map_err(|e| e.to_string())?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        }).map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?
    };

    info!("[OBJECT_INDEX] Starting to index {} images", images.len());

    let mut indexed = 0;
    let mut failed = 0;

    for (image_id, image_path) in &images {
        // Check if file exists
        if !std::path::Path::new(&image_path).exists() {
            warn!("[OBJECT_INDEX] Image not found: {}", image_path);
            failed += 1;
            continue;
        }

        // Use ObjectSearch's ROI extraction and embedding
        match searcher.extract_rois(&image_path) {
            Ok(rois) => {
                if rois.is_empty() {
                    continue;
                }

                // For each ROI, create embedding and add to HNSW
                let img = match image::open(&image_path) {
                    Ok(i) => i.to_rgb8(),
                    Err(e) => {
                        warn!("[OBJECT_INDEX] Failed to open {}: {}", image_path, e);
                        failed += 1;
                        continue;
                    }
                };

                for roi in &rois {
                    let x1 = roi.bbox[0] as u32;
                    let y1 = roi.bbox[1] as u32;
                    let x2 = (x1 + roi.bbox[2] as u32).min(img.width());
                    let y2 = (y1 + roi.bbox[3] as u32).min(img.height());

                    if x2 <= x1 || y2 <= y1 {
                        continue;
                    }

                    let crop = DynamicImage::ImageRgb8(img.clone()).crop_imm(x1, y1, x2 - x1, y2 - y1);
                    match searcher.extract_embedding(&crop) {
                        Ok(embedding) => {
                            // Add to category index
                            if let Err(e) = searcher.add_to_index(*image_id, &image_path, &roi.region_type, roi.bbox, &embedding) {
                                warn!("[OBJECT_INDEX] Failed to add to index: {}", e);
                            } else {
                                indexed += 1;
                            }
                        }
                        Err(e) => {
                            warn!("[OBJECT_INDEX] Embedding failed: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                warn!("[OBJECT_INDEX] ROI extraction failed for {}: {}", image_path, e);
                failed += 1;
            }
        }

        // Log progress every 100 images
        if (indexed + failed) % 100 == 0 {
            info!("[OBJECT_INDEX] Progress: {} indexed, {} failed", indexed, failed);
        }
    }

    // Save the index
    if let Err(e) = searcher.save_index() {
        warn!("[OBJECT_INDEX] Failed to save index: {}", e);
    }

    info!("[OBJECT_INDEX] Complete: {} indexed, {} failed", indexed, failed);
    Ok(format!("Indexed {} images ({} failed)", indexed, failed))
}

#[tauri::command]
pub async fn copy_files(source_paths: Vec<String>, dest_dir: String) -> Result<Vec<String>, String> {
    let dest_path = std::path::PathBuf::from(&dest_dir);

    // Ensure destination directory exists
    std::fs::create_dir_all(&dest_path).map_err(|e| format!("Failed to create destination directory: {}", e))?;

    let mut results = Vec::new();
    let mut errors = Vec::new();

    for source in &source_paths {
        let source_path = std::path::PathBuf::from(source);

        // Get filename
        let filename = match source_path.file_name() {
            Some(name) => name,
            None => {
                errors.push(format!("Invalid filename: {}", source));
                continue;
            }
        };

        let dest_file = dest_path.join(filename);

        match std::fs::copy(&source_path, &dest_file) {
            Ok(_) => {
                let dest_str = dest_file.to_string_lossy().to_string();
                info!("[COPY] Copied: {} -> {}", source, dest_str);
                results.push(dest_str);
            }
            Err(e) => {
                warn!("[COPY] Failed to copy {}: {}", source, e);
                errors.push(format!("{}: {}", source, e));
            }
        }
    }

    if results.is_empty() && !errors.is_empty() {
        return Err(format!("Failed to copy any files: {}", errors.join(", ")));
    }

    Ok(results)
}

