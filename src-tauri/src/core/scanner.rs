use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs;
use std::time::UNIX_EPOCH;
use anyhow::Result;
use tracing::{info, warn};
use tokio::task;
use tokio::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::database::{Database, images::ImagesTable, tasks::TasksTable};
use crate::tauri_cmd::commands::ScanStats;
use image::GenericImageView;

const BATCH_SIZE: usize = 500;
const YIELD_EVERY: usize = 50;

pub struct Scanner {
    db: Arc<Database>,
}

impl Scanner {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    pub async fn scan_folder(&self, folder_path: &Path) -> Result<ScanResult> {
        self.scan_folder_with_progress(folder_path, None).await
    }

    pub async fn scan_folder_with_progress(&self, folder_path: &Path, progress_stats: Option<Arc<RwLock<ScanStats>>>) -> Result<ScanResult> {
        info!("Starting folder scan: {:?}", folder_path);

        let mut all_image_paths = Vec::new();

        for entry in walkdir::WalkDir::new(folder_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if Self::is_image_file(path) {
                all_image_paths.push(path.to_path_buf());
            }
        }

        info!("Found {} image files", all_image_paths.len());

        let mut new_images = 0;
        let mut skipped = 0;
        let mut batch_paths = Vec::with_capacity(BATCH_SIZE);
        let mut batch_hashes = Vec::with_capacity(BATCH_SIZE);
        let mut batch_sizes = Vec::with_capacity(BATCH_SIZE);
        let mut batch_mtimes = Vec::with_capacity(BATCH_SIZE);
        let mut processed = 0;
        let total_images = all_image_paths.len();

        for path in &all_image_paths {
            match self.process_single(path) {
                Ok((path_str, hash, size, mtime, is_new)) => {
                    batch_paths.push(path_str);
                    batch_hashes.push(hash);
                    batch_sizes.push(size);
                    batch_mtimes.push(mtime);
                    if is_new {
                        new_images += 1;
                    } else {
                        skipped += 1;
                    }
                    if batch_paths.len() >= BATCH_SIZE {
                        self.flush_batch(&batch_paths, &batch_hashes, &batch_sizes, &batch_mtimes)?;
                        batch_paths.clear();
                        batch_hashes.clear();
                        batch_sizes.clear();
                        batch_mtimes.clear();

                        // Yield to tokio runtime to allow other tasks (including UI updates)
                        task::yield_now().await;
                    }
                }
                Err(e) => {
                    warn!("Failed to process {}: {}", path.display(), e);
                }
            }

            processed += 1;

            // Update progress stats
            if processed % YIELD_EVERY == 0 {
                if let Some(stats) = &progress_stats {
                    let mut s = stats.write().await;
                    s.processed_images = processed;
                    s.total_images = total_images;
                    s.current_file = path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                }
                task::yield_now().await;
            }
        }

        // Final progress update
        if let Some(stats) = &progress_stats {
            let mut s = stats.write().await;
            s.processed_images = processed;
            s.total_images = total_images;
            s.current_file = String::new();
        }

        if !batch_paths.is_empty() {
            self.flush_batch(&batch_paths, &batch_hashes, &batch_sizes, &batch_mtimes)?;
        }

        info!("Scan complete: {} new, {} skipped", new_images, skipped);

        Ok(ScanResult {
            total_found: all_image_paths.len(),
            new_images,
            skipped,
        })
    }

    fn process_single(&self, path: &Path) -> Result<(String, String, i64, i64, bool)> {
        let metadata = fs::metadata(path)?;
        let size = metadata.len() as i64;
        let modified_time = metadata.modified()?
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        if let Some(existing_hash) = ImagesTable::check_file_signature(
            &self.db.conn,
            &path.to_string_lossy(),
            size,
            modified_time
        )? {
            return Ok((path.to_string_lossy().to_string(), existing_hash, size, modified_time, false));
        }

        let hash = Self::compute_hash(path)?;

        if ImagesTable::exists_by_hash(&self.db.conn, &hash)? {
            ImagesTable::update_hash(&self.db.conn, &path.to_string_lossy(), &hash, size, modified_time)?;
            return Ok((path.to_string_lossy().to_string(), hash, size, modified_time, false));
        }

        Ok((path.to_string_lossy().to_string(), hash, size, modified_time, true))
    }

    /// Filter non-photo images (vector graphics, icons, screenshots, etc.)
    /// Returns true if the image should be skipped for face pipeline
    fn should_skip_for_face_pipeline(path: &Path, file_size: u64) -> bool {
        // Try to open and analyze the image
        if let Ok(img) = image::open(path) {
            let (width, height) = img.dimensions();
            let pixels = (width * height) as f32;

            // bytes_per_pixel = file_size / pixels
            // Real photos: typically 0.2-3.0 bytes/pixel (compressed)
            // Vector graphics/icons: typically < 0.1 bytes/pixel
            let bpp = file_size as f32 / pixels;

            if bpp < 0.05 {
                info!("Skipping non-photo image: {:?} (bpp={:.3}, size={}x{})", path, bpp, width, height);
                return true;
            }

            // Check unique colors (optional secondary filter)
            // This is slow, so we only do it for borderline cases
            if bpp < 0.15 {
                if let Some(unique_colors) = Self::count_unique_colors(&img) {
                    if unique_colors < 100 {
                        info!("Skipping low-color image: {:?} (colors={})", path, unique_colors);
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Count approximate unique colors in image (sampled for performance)
    fn count_unique_colors(img: &image::DynamicImage) -> Option<usize> {
        let img = img.resize(50, 50, image::imageops::FilterType::Nearest);
        let rgba = img.to_rgba8();
        let mut unique: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for pixel in rgba.pixels() {
            // Pack RGB into u32 (ignore alpha for color counting)
            let rgb = ((pixel[0] as u32) << 16) | ((pixel[1] as u32) << 8) | (pixel[2] as u32);
            unique.insert(rgb);
            if unique.len() > 10000 {
                // Real photos will have more than 10000 unique colors
                return Some(unique.len());
            }
        }

        Some(unique.len())
    }

    fn flush_batch(&self, paths: &[String], hashes: &[String], sizes: &[i64], mtimes: &[i64]) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let mut conn = self.db.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let now = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let mut image_ids = Vec::new();
        for i in 0..paths.len() {
            // Check if image already exists by path
            let existing: Option<i64> = tx.query_row(
                "SELECT id FROM images WHERE path = ?1",
                rusqlite::params![paths[i]],
                |row| row.get(0),
            ).ok();

            if let Some(image_id) = existing {
                // Image exists, check if it needs processing (scan_status = 'pending')
                let needs_processing: bool = tx.query_row(
                    "SELECT scan_status = 'pending' FROM images WHERE id = ?1",
                    rusqlite::params![image_id],
                    |row| row.get(0),
                ).unwrap_or(false);

                if needs_processing {
                    image_ids.push(image_id);
                }
            } else {
                // New image, insert it
                tx.execute(
                    "INSERT INTO images (path, hash, size, modified_time, width, height, thumbnail_status, scan_status, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, NULL, NULL, 'pending', 'pending', ?5, ?6)",
                    rusqlite::params![paths[i], hashes[i], sizes[i], mtimes[i], now, now],
                )?;
                let image_id = tx.last_insert_rowid();
                image_ids.push(image_id);
            }
        }

        for id in &image_ids {
            tx.execute(
                "INSERT INTO scan_tasks (image_id, task_type, status, retry_count, created_at) VALUES (?1, 'image', 'pending', 0, ?2)",
                rusqlite::params![id, now],
            )?;
        }

        tx.commit()?;

        Ok(())
    }

    fn is_image_file(path: &Path) -> bool {
        if !path.is_file() {
            return false;
        }
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => {
                let ext = ext.to_lowercase();
                matches!(ext.as_str(),
                    "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tiff" | "tif" |
                    "heic" | "heif" | "avif" | "svg" | "ico" | "raw" | "cr2" | "nef" |
                    "arw" | "dng" | "orf" | "rw2" | "pef" | "srw" | "x3f" | "3fr" |
                    "raf" | "mrw" | "nrw" | "dcr" | "mos" | "crw" | "erf" | "mdc"
                )
            }
            None => false,
        }
    }

    fn compute_hash(path: &Path) -> Result<String> {
        let content = fs::read(path)?;
        let hash = blake3::hash(&content);
        Ok(hash.to_hex().to_string())
    }
}

#[derive(Debug)]
pub struct ScanResult {
    pub total_found: usize,
    pub new_images: usize,
    pub skipped: usize,
}