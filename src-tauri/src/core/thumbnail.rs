use anyhow::Result;
use std::path::{Path, PathBuf};
use std::fs;
use std::sync::Arc;
use std::sync::RwLock;
use image::{ImageDecoder, GenericImageView};
use tracing::{info, warn};

const THUMBNAIL_MAX_SIZE: u32 = 256;
const THUMBNAIL_QUALITY: u8 = 85;

pub struct ThumbnailStore {
    base_dir: PathBuf,
}

impl ThumbnailStore {
    pub fn new(data_dir: &Path) -> Self {
        let base_dir = data_dir.join("thumbnails");
        Self { base_dir }
    }

    pub fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.base_dir)?;
        Ok(())
    }

    pub fn base_dir(&self) -> &PathBuf {
        &self.base_dir
    }

    pub fn get_thumbnail_path(&self, hash: &str) -> PathBuf {
        let prefix = &hash[..2];
        let filename = format!("{}.jpg", hash);
        self.base_dir.join(prefix).join(filename)
    }

    pub fn generate(&self, image_path: &str, hash: &str) -> Result<String> {
        let thumb_path = self.get_thumbnail_path(hash);

        if thumb_path.exists() {
            return Ok(thumb_path.to_string_lossy().to_string());
        }

        if let Some(parent) = thumb_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let img = image::open(image_path)?;
        let (width, height) = img.dimensions();

        let scale = if width > height {
            THUMBNAIL_MAX_SIZE as f32 / width as f32
        } else {
            THUMBNAIL_MAX_SIZE as f32 / height as f32
        };

        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        let thumbnail = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);

        let mut output = std::io::BufWriter::new(
            fs::File::create(&thumb_path)?
        );

        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
            &mut output,
            THUMBNAIL_QUALITY
        );

        thumbnail.write_with_encoder(encoder)?;

        info!("Generated thumbnail: {}", thumb_path.display());

        Ok(thumb_path.to_string_lossy().to_string())
    }

    pub fn exists(&self, hash: &str) -> bool {
        self.get_thumbnail_path(hash).exists()
    }

    pub fn delete(&self, hash: &str) -> Result<()> {
        let path = self.get_thumbnail_path(hash);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    pub fn calculate_size(&self) -> u64 {
        if !self.base_dir.exists() {
            return 0;
        }

        std::fs::read_dir(&self.base_dir)
            .map(|entries| {
                entries.filter_map(|e| e.ok())
                    .flat_map(|e| std::fs::read_dir(e.path()).map(|i| i.filter_map(|f| f.ok())))
                    .flatten()
                    .filter(|e| e.path().extension().map(|ext| ext == "jpg").unwrap_or(false))
                    .map(|e| e.path().metadata().map(|m| m.len()).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn count(&self) -> usize {
        if !self.base_dir.exists() {
            return 0;
        }

        std::fs::read_dir(&self.base_dir)
            .map(|entries| {
                entries.filter_map(|e| e.ok())
                    .flat_map(|e| std::fs::read_dir(e.path()).map(|i| i.filter_map(|f| f.ok())))
                    .flatten()
                    .filter(|e| e.path().extension().map(|ext| ext == "jpg").unwrap_or(false))
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn clear_all(&self) -> Result<()> {
        if !self.base_dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                std::fs::remove_dir_all(&path)?;
            } else {
                std::fs::remove_file(&path)?;
            }
        }
        info!("Cleared all thumbnails");
        Ok(())
    }
}

pub struct ThumbnailWorker {
    store: Arc<ThumbnailStore>,
    thread_pool: rayon::ThreadPool,
}

impl ThumbnailWorker {
    pub fn new(num_threads: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        Self {
            store: Arc::new(ThumbnailStore::new(&std::path::PathBuf::from("."))),
            thread_pool,
        }
    }

    pub fn with_store(store: Arc<ThumbnailStore>, num_threads: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        Self { store, thread_pool }
    }

    pub fn generate_thumbnail(&self, image_path: &str, hash: &str) -> Result<String> {
        self.store.generate(image_path, hash)
    }

    pub fn generate_batch(&self, tasks: Vec<(String, String)>) -> Vec<(String, Result<String>)> {
        use rayon::prelude::*;

        tasks.into_par_iter()
            .map(|(image_path, hash)| {
                let result = self.store.generate(&image_path, &hash);
                (hash, result)
            })
            .collect()
    }

    pub fn exists(&self, hash: &str) -> bool {
        self.store.exists(hash)
    }

    pub fn delete(&self, hash: &str) -> Result<()> {
        self.store.delete(hash)
    }

    pub fn count(&self) -> usize {
        self.store.count()
    }

    pub fn size(&self) -> u64 {
        self.store.calculate_size()
    }
}