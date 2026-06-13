pub mod core;
pub mod ai;
pub mod search;
pub mod tauri_cmd;
pub mod category_search;

use std::sync::Arc;
use std::path::PathBuf;
use tracing::{info, error};
use tracing_subscriber::prelude::*;

use core::Database;
use core::statistics::StatisticsCollector;
use core::thumbnail::ThumbnailStore;
use core::scanner::Scanner;
use core::start_processing_service;
use search::person_search::PersonSearch;
use tokio::sync::RwLock;

pub fn run() {
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("PhotoFinderNext")
        .join("logs");
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = tracing_appender::rolling::daily(&log_dir, "photofinder.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking))
        .with(tracing_subscriber::EnvFilter::new("info"))
        .init();

    info!("PhotoFinder Next starting...");

    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("PhotoFinderNext");

    eprintln!("[MAIN] data_dir = {}", data_dir.display());

    let db_path = data_dir.join("photofinder.db");
    std::fs::create_dir_all(db_path.parent().unwrap()).ok();
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "[MAIN] db_path = {}", db_path.display());
    let _ = writeln!(std::io::stderr(), "[MAIN] About to call Database::new...");
    std::io::stderr().flush().ok();

    let db = match Database::new(&db_path) {
        Ok(db) => {
            use std::io::Write;
            let _ = writeln!(std::io::stderr(), "[MAIN] Database::new succeeded!");
            std::io::stderr().flush().ok();
            Arc::new(db)
        }
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("[MAIN] db created");

    eprintln!("[MAIN] Creating stats...");
    let stats = StatisticsCollector::new(Arc::clone(&db), &data_dir);
    eprintln!("[MAIN] stats created");

    eprintln!("[MAIN] Creating thumbnail_store...");
    let thumbnail_store = Arc::new(ThumbnailStore::new(&data_dir));
    eprintln!("[MAIN] thumbnail_store created");

    eprintln!("[MAIN] Creating scanner...");
    let scanner = Arc::new(Scanner::new(Arc::clone(&db)));
    eprintln!("[MAIN] scanner created");

    // Start background processing service
    eprintln!("[MAIN] About to call start_processing_service...");
    let processing_stats = Arc::new(RwLock::new(tauri_cmd::commands::ProcessingStats::default()));
    start_processing_service(Arc::clone(&db), data_dir.clone(), Arc::clone(&processing_stats));
    eprintln!("[MAIN] start_processing_service returned!");

    // Initialize search modules
    eprintln!("[MAIN] Creating person_search...");
    let person_search = match PersonSearch::new(Arc::clone(&db), &data_dir) {
        Ok(ps) => Arc::new(RwLock::new(ps)),
        Err(e) => {
            eprintln!("[MAIN] Failed to create person search: {}", e);
            error!("Failed to create person search: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("[MAIN] person_search created");

    let app_state = tauri_cmd::commands::AppState {
        db,
        stats,
        thumbnail_store,
        scanner,
        person_search,
        object_search: Arc::new(RwLock::new(None)),
        category_search: Arc::new(RwLock::new(None)),
        is_scanning: Arc::new(RwLock::new(false)),
        is_processing: Arc::new(RwLock::new(false)),
        last_scan_stats: Arc::new(RwLock::new(tauri_cmd::commands::ScanStats::default())),
        processing_stats,
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            tauri_cmd::commands::scan_folder,
            tauri_cmd::commands::stop_scan,
            tauri_cmd::commands::clear_database,
            tauri_cmd::commands::rebuild_thumbnails,
            tauri_cmd::commands::search_person,
            tauri_cmd::commands::get_scan_status,
            tauri_cmd::commands::get_processing_status,
            tauri_cmd::commands::get_statistics,
            tauri_cmd::commands::test_face_pipeline,
            tauri_cmd::commands::test_face_similarity,
            tauri_cmd::commands::write_query_image,
            tauri_cmd::commands::write_cropped_image,
            tauri_cmd::commands::rebuild_face_index,
            tauri_cmd::commands::debug_faces,
            tauri_cmd::commands::init_object_search,
            tauri_cmd::commands::search_objects,
            tauri_cmd::commands::index_images_for_object,
            tauri_cmd::commands::get_image_thumbnail,
            tauri_cmd::commands::copy_files,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}