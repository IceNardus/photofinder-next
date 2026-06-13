//! Test script for patch search
//! Run with: cargo test --package photofinder-next --lib search::test_patch_search -- --nocapture

use crate::search::PatchSearch;
use std::sync::Arc;
use crate::core::database::Database;
use crate::core::features::PatchConfig;
use tracing::{info, error};

/// Test patch search with a specific image
pub fn test_search_with_image(
    query_path: &str,
    top_k: usize,
) -> Result<Vec<crate::search::PatchSearchResult>, String> {
    // Initialize database
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data local dir".to_string())?;
    let db_path = data_dir.join("photofinder.db");
    let db = Database::new(&db_path)
        .map_err(|e| format!("Failed to open database: {}", e))?;

    // Initialize patch search
    let superpoint_path = "/Users/mac/Library/Caches/PhotoFinder/models-export/superpoint.onnx";
    let lightglue_path = "/Users/mac/Library/Caches/PhotoFinder/models-export/lightglue_onnx/weights/lightglue.onnx";

    let config = crate::search::PatchSearchConfig::default();
    let searcher = PatchSearch::new(
        db,
        superpoint_path,
        lightglue_path,
        config,
    ).map_err(|e| format!("Failed to create searcher: {}", e))?;

    // Run search
    info!("[TEST] Running search for: {}", query_path);
    let results = searcher.search(query_path, top_k)?;

    info!("[TEST] Found {} results", results.len());
    for (i, r) in results.iter().take(10).enumerate() {
        info!("[TEST] Result {}: image_id={}, inliers={}, confidence={:.2}, path={}",
              i + 1, r.image_id, r.inlier_count, r.confidence, r.image_path);
    }

    Ok(results)
}

/// Test search and print detailed debug info
pub fn test_search_debug(query_path: &str) -> Result<(), String> {
    use std::path::Path;

    info!("[DEBUG] Starting debug search test");
    info!("[DEBUG] Query path: {}", query_path);
    info!("[DEBUG] Query exists: {}", Path::new(query_path).exists());

    // Check database
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| "Failed to get data local dir".to_string())?;
    let db_path = data_dir.join("photofinder.db");
    info!("[DEBUG] DB path: {:?}", db_path);

    let db = Database::new(&db_path)
        .map_err(|e| format!("Failed to open database: {}", e))?;

    // Count patches
    let conn = db.conn.lock().unwrap();
    let patch_count: i64 = rusqlite::params!(
        "SELECT COUNT(*) FROM patch_vectors"
    ).query_row(&conn, [], |row| row.get(0)).unwrap();
    info!("[DEBUG] Total patch vectors in DB: {}", patch_count);

    let image_count: i64 = rusqlite::params!(
        "SELECT COUNT(DISTINCT image_id) FROM patch_features"
    ).query_row(&conn, [], |row| row.get(0)).unwrap();
    info!("[DEBUG] Unique images with patches: {}", image_count);

    drop(conn);

    // Run the actual search
    let results = test_search_with_image(query_path, 10)?;

    info!("[DEBUG] Search completed, {} results", results.len());
    for r in results.iter().take(5) {
        eprintln!("Result: {} inliers, {} matches, conf={:.2}", r.inlier_count, r.total_matches, r.confidence);
    }

    Ok(())
}