//! Standalone test binary for patch search
//! Run with: cargo run --example test_search

use std::path::Path;
use photofinder_next_lib::search::PatchSearch;
use photofinder_next_lib::search::PatchSearchConfig;
use photofinder_next_lib::core::Database;

fn main() {
    println!("=== Patch Search Test ===\n");

    // Use the same path construction as the main app
    let data_dir = dirs::data_local_dir()
        .unwrap()
        .join("PhotoFinderNext");

    // Check query image
    let query_path = data_dir.join("query_image.png");
    println!("Query image: {:?}", query_path);
    println!("Query image exists: {}", Path::new(&query_path).exists());

    // DB path
    let db_path = data_dir.join("photofinder.db");
    println!("\nDB path: {:?}", db_path);
    println!("DB exists: {}", Path::new(&db_path).exists());

    // Initialize database
    println!("\nInitializing database...");
    let db = match Database::new(&db_path) {
        Ok(db) => db,
        Err(e) => {
            println!("Failed to open database: {}", e);
            return;
        }
    };

    // Count patches
    let conn = db.conn.lock().unwrap();
    let patch_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM patch_vectors", [], |row| row.get(0)
    ).unwrap();
    let image_count: i64 = conn.query_row(
        "SELECT COUNT(DISTINCT image_id) FROM patch_features", [], |row| row.get(0)
    ).unwrap();
    drop(conn);

    println!("Patch vectors in DB: {}", patch_count);
    println!("Images with patches: {}", image_count);

    if patch_count == 0 {
        println!("\nERROR: No patch vectors in database! Cannot search.");
        println!("Please index some images first using the app.");
        return;
    }

    // Initialize patch search
    println!("\nInitializing patch search...");
    let cache_dir = dirs::cache_dir().unwrap().join("PhotoFinder");
    let superpoint_path = cache_dir.join("models-export").join("superpoint.onnx");
    let lightglue_path = cache_dir.join("models-export").join("lightglue_onnx").join("weights").join("lightglue.onnx");

    println!("SuperPoint: {:?}", superpoint_path);
    println!("LightGlue: {:?}", lightglue_path);

    let mut config = PatchSearchConfig::default();
    config.data_dir = Some(data_dir.join("index").to_string_lossy().to_string());

    let searcher = match PatchSearch::new(
        db,
        superpoint_path.to_str().unwrap(),
        lightglue_path.to_str().unwrap(),
        config,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("Failed to create searcher: {}", e);
            return;
        }
    };
    println!("Patch search initialized!");

    // Rebuild index to populate id_to_patch
    println!("\nRebuilding patch index...");
    match searcher.rebuild_index() {
        Ok(_) => println!("Index rebuilt successfully"),
        Err(e) => println!("Failed to rebuild index: {}", e)
    }

    // Run search
    println!("\nRunning search for: {:?}", query_path);

    // First extract features to see what we get
    println!("\nExtracting query features...");
    match searcher.extract_query_features(query_path.to_str().unwrap()) {
        Ok(features) => {
            println!("Extracted {} patches from query image", features.patches.len());
            println!("Image size: {}x{}", features.width, features.height);
            for (i, p) in features.patches.iter().take(5).enumerate() {
                println!("  Patch {}: {} keypoints, bbox={:.2},{:.2},{:.2},{:.2}",
                    i, p.num_keypoints, p.bbox.x, p.bbox.y, p.bbox.w, p.bbox.h);
            }
            if features.patches.len() > 5 {
                println!("  ... and {} more patches", features.patches.len() - 5);
            }
        },
        Err(e) => {
            println!("Feature extraction failed: {}", e);
            return;
        }
    }

    println!("\nRunning search...");
    let search_result = searcher.search(query_path.to_str().unwrap(), 10);
    match search_result {
        Ok(results) => {
            println!("\n=== Search Results ===");
            println!("Found {} results\n", results.len());
            for (i, r) in results.iter().enumerate() {
                println!("{}. image_id={}, inliers={}, matches={}, conf={:.2}", i + 1, r.image_id, r.inlier_count, r.total_matches, r.confidence);
                println!("   path={}", r.image_path);
            }
        },
        Err(e) => {
            println!("Search failed: {}", e);
        }
    }
}