//! Index for category object search using HNSW + fine-grained matching

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::core::index::hnsw::HnswIndex;

/// Embedding metadata stored in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    pub id: i64,
    pub image_id: i64,
    pub image_path: String,
    pub class_name: String,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

/// Index with HNSW for coarse search
pub struct CategoryIndex {
    vectors: Vec<Vec<f32>>,
    ids: Vec<i64>,
    hnsw: HnswIndex,
    conn: Arc<Mutex<Connection>>,
}

// Safety: CategoryIndex wraps Connection in Arc<Mutex>, which ensures thread-safe access
unsafe impl Send for CategoryIndex {}
unsafe impl Sync for CategoryIndex {}

impl CategoryIndex {
    /// Create a new index or load existing one
    pub fn new(db_path: &Path) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        // Create tables if not exists
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS category_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                confidence REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS category_embeddings (
                id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (id) REFERENCES category_objects(id)
            );

            CREATE INDEX IF NOT EXISTS idx_category_objects_image_id ON category_objects(image_id);
            CREATE INDEX IF NOT EXISTS idx_category_objects_class ON category_objects(class_name);
        "#).map_err(|e| format!("Failed to create tables: {}", e))?;

        Ok(Self {
            vectors: Vec::new(),
            ids: Vec::new(),
            hnsw: HnswIndex::new(512),  // MobileCLIP outputs 512-dim vectors
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Load existing index from database
    #[allow(dead_code)]
    pub fn load(db_path: &Path) -> Result<Self, String> {
        let mut index = Self::new(db_path)?;
        index.load_from_db()?;
        Ok(index)
    }

    /// Load vectors from database
    fn load_from_db(&mut self) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, vector FROM category_embeddings")
            .map_err(|e| format!("Failed to prepare: {}", e))?;

        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, blob))
        }).map_err(|e| format!("Failed to query: {}", e))?;

        for row in rows {
            if let Ok((id, blob)) = row {
                let vector = bytes_to_f32(&blob);
                self.ids.push(id);
                self.vectors.push(vector);
            }
        }

        Ok(())
    }

    /// Add an object embedding to the index
    #[allow(dead_code)]
    pub fn add_object(
        &mut self,
        image_id: i64,
        image_path: &str,
        class_name: &str,
        bbox: (f32, f32, f32, f32),
        confidence: f32,
        embedding: &[f32],
    ) -> Result<i64, String> {
        let (x1, y1, x2, y2) = bbox;

        // Insert metadata into SQLite
        let id = {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                r#"
                INSERT INTO category_objects (image_id, image_path, class_name, x1, y1, x2, y2, confidence)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                "#,
                params![image_id, image_path, class_name, x1, y1, x2, y2, confidence],
            ).map_err(|e| format!("Failed to insert metadata: {}", e))?;

            conn.last_insert_rowid()
        };

        // Insert vector
        let vector_bytes = f32_to_bytes(embedding);
        {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO category_embeddings (id, vector) VALUES (?1, ?2)",
                params![id, vector_bytes],
            ).map_err(|e| format!("Failed to insert vector: {}", e))?;
        }

        // Keep in-memory copy and add to HNSW
        self.ids.push(id);
        self.vectors.push(embedding.to_vec());
        self.hnsw.add(embedding.to_vec(), id);

        Ok(id)
    }

    /// Search for similar objects using HNSW
    #[allow(dead_code)]
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(ObjectMetadata, f32)>, String> {
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }

        // Use HNSW for fast search
        let hnsw_results = self.hnsw.search(query, k);

        let mut results = Vec::new();
        for hnsw_result in hnsw_results {
            if let Some(meta) = self.get_object_by_id(hnsw_result.id)? {
                results.push((meta, hnsw_result.score));
            }
        }

        Ok(results)
    }

    /// Search and return raw HNSW results (for fine-grained matching)
    pub fn search_hnsw(&self, query: &[f32], k: usize) -> Vec<crate::core::index::hnsw::SearchResult> {
        self.hnsw.search(query, k)
    }

    /// Get object metadata by ID
    pub fn get_object_by_id(&self, id: i64) -> Result<Option<ObjectMetadata>, String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, image_path, class_name, x1, y1, x2, y2, confidence FROM category_objects WHERE id = ?1"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let result = stmt.query_row(params![id], |row| {
            let class_name: Option<String> = row.get(3)?;
            Ok(ObjectMetadata {
                id: row.get(0)?,
                image_id: row.get(1)?,
                image_path: row.get(2)?,
                class_name: class_name.unwrap_or_else(|| "unknown".to_string()),
                x1: row.get(4)?,
                y1: row.get(5)?,
                x2: row.get(6)?,
                y2: row.get(7)?,
                confidence: row.get(8)?,
            })
        });

        match result {
            Ok(meta) => Ok(Some(meta)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Failed to query metadata: {}", e)),
        }
    }

    /// Get total number of objects indexed
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Clear all data
    #[allow(dead_code)]
    pub fn clear(&mut self) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM category_embeddings", [])
            .map_err(|e| format!("Failed to clear embeddings: {}", e))?;
        conn.execute("DELETE FROM category_objects", [])
            .map_err(|e| format!("Failed to clear objects: {}", e))?;

        drop(conn);

        self.vectors.clear();
        self.ids.clear();

        Ok(())
    }
}

/// Compute cosine similarity between two vectors
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 1e-8 && norm_b > 1e-8 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Convert f32 vector to bytes
fn f32_to_bytes(vec: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for &v in vec {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Convert bytes to f32 vector
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut vec = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks(4) {
        if chunk.len() == 4 {
            vec.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
    }
    vec
}