use std::sync::Arc;
use std::path::Path;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::core::database::Database;
use crate::core::database::images::ImagesTable;
use crate::core::database::faces::FacesTable;
use crate::core::database::tasks::TasksTable;
use crate::core::thumbnail::ThumbnailStore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub image_count: i64,
    pub face_count: i64,
    pub pending_task_count: i64,
    pub index_size_bytes: u64,
    pub vector_store_size_bytes: u64,
    pub database_size_bytes: u64,
    pub thumbnail_count: usize,
    pub thumbnail_size_bytes: u64,
}

pub struct StatisticsCollector {
    db: Arc<Database>,
    thumbnail_store: Arc<ThumbnailStore>,
}

impl StatisticsCollector {
    pub fn new(db: Arc<Database>, data_dir: &Path) -> Self {
        let thumbnail_store = Arc::new(ThumbnailStore::new(data_dir));
        Self { db, thumbnail_store }
    }

    pub async fn collect(&self, data_dir: &Path, db_path: &Path) -> Result<Statistics> {
        let conn = &self.db.conn;

        let image_count = ImagesTable::count(conn)?;
        let face_count = FacesTable::count(conn)?;
        let pending_task_count = TasksTable::pending_count(conn)?;
        let thumbnail_count = self.thumbnail_store.count();
        let thumbnail_size_bytes = self.thumbnail_store.calculate_size();

        let index_size_bytes = self.calculate_dir_size(&data_dir.join("index"));
        let vector_store_size_bytes = self.calculate_dir_size(&data_dir.join("vectors"));
        let database_size_bytes = db_path.metadata().map(|m| m.len()).unwrap_or(0);

        Ok(Statistics {
            image_count,
            face_count,
            pending_task_count,
            index_size_bytes,
            vector_store_size_bytes,
            database_size_bytes,
            thumbnail_count,
            thumbnail_size_bytes,
        })
    }

    pub fn thumbnail_generation_speed(&self, elapsed_secs: f64, generated: usize) -> f64 {
        if elapsed_secs > 0.0 {
            generated as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    fn calculate_dir_size(&self, path: &Path) -> u64 {
        if !path.exists() {
            return 0;
        }
        std::fs::read_dir(path)
            .map(|entries| {
                entries.filter_map(|e| e.ok())
                    .map(|e| {
                        if e.path().is_file() {
                            e.path().metadata().map(|m| m.len()).unwrap_or(0)
                        } else {
                            0
                        }
                    })
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn format_size(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}