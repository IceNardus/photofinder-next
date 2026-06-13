//! Object HNSW Index - Wrapper for HNSW index with object-level search
//!
//! Key design:
//! - Each node represents an object (ROI), not an image
//! - Search returns Top-N objects
//! - Objects are aggregated by image_id for final results

use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, anyhow};
use tracing::info;

use crate::core::index::hnsw::HnswIndex;
use super::vector_store::{VectorStore, VectorMeta};

const VECTOR_DIM: usize = 512;

#[derive(Debug, Clone)]
pub struct ObjectSearchResult {
    pub image_id: i64,
    pub image_path: String,
    pub score: f32,
    pub bbox: (f32, f32, f32, f32),
    pub region_type: String,
}

pub struct ObjectHnswIndex {
    hnsw: HnswIndex,
    vector_store: VectorStore,
    id_to_meta: HashMap<i64, VectorMeta>,  // object_id -> meta
}

impl ObjectHnswIndex {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let vector_store = VectorStore::new(data_dir)?;
        let hnsw = HnswIndex::new(VECTOR_DIM);

        Ok(Self {
            hnsw,
            vector_store,
            id_to_meta: HashMap::new(),
        })
    }

    /// Load existing index from disk
    #[allow(dead_code)]
    pub fn load(&mut self) -> Result<()> {
        // Load vectors and rebuild HNSW
        let (embeddings, metas) = self.vector_store.load_all()?;

        eprintln!("[ObjectHnsw] load_all returned: {} embeddings, {} metas", embeddings.len(), metas.len());

        for (i, meta) in metas.iter().enumerate() {
            if i < embeddings.len() {
                self.hnsw.add(embeddings[i].clone(), meta.object_id);
                self.id_to_meta.insert(meta.object_id, meta.clone());
            }
        }

        eprintln!("[ObjectHnsw] Loaded {} objects into HNSW index", self.id_to_meta.len());
        info!("[ObjectHnsw] Loaded {} objects from index", self.id_to_meta.len());
        Ok(())
    }

    /// Add an object embedding
    pub fn add_object(
        &mut self,
        image_id: i64,
        image_path: &str,
        region_type: &str,
        bbox: (f32, f32, f32, f32),
        embedding: &[f32],
    ) -> Result<i64> {
        let object_id = self.vector_store.count() as i64 + 1;

        let offset = self.vector_store.add(
            object_id,
            image_id,
            image_path,
            bbox,
            region_type,
            embedding,
        )?;

        self.hnsw.add(embedding.to_vec(), object_id);

        let meta = VectorMeta {
            object_id,
            offset,
            image_id,
            image_path: image_path.to_string(),
            bbox,
            region_type: region_type.to_string(),
        };
        self.id_to_meta.insert(object_id, meta);

        Ok(object_id)
    }

    /// Search top-N objects, then aggregate by image
    pub fn search(&self, query_embedding: &[f32], top_k_objects: usize) -> Vec<ObjectSearchResult> {
        // Step 1: HNSW search for top objects
        let object_results = self.hnsw.search(query_embedding, top_k_objects * 3); // Over-fetch for aggregation

        // Step 2: Aggregate by image_id (max score per image)
        let mut image_scores: HashMap<i64, (f32, &VectorMeta)> = HashMap::new();

        for result in object_results {
            if let Some(meta) = self.id_to_meta.get(&result.id) {
                let current = image_scores.entry(meta.image_id).or_insert((0.0, meta));

                // Keep max score
                if result.score > current.0 {
                    *current = (result.score, meta);
                }
            }
        }

        // Step 3: Convert to sorted results
        let mut results: Vec<ObjectSearchResult> = image_scores.into_iter()
            .map(|(image_id, (score, meta))| ObjectSearchResult {
                image_id,
                image_path: meta.image_path.clone(),
                score,
                bbox: meta.bbox,
                region_type: meta.region_type.clone(),
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        results.truncate(top_k_objects);
        results
    }

    #[allow(dead_code)]
    pub fn object_count(&self) -> usize {
        self.id_to_meta.len()
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) -> Result<()> {
        self.vector_store.clear()?;
        self.hnsw = HnswIndex::new(VECTOR_DIM);
        self.id_to_meta.clear();
        Ok(())
    }
}