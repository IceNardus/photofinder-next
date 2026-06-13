use anyhow::Result;
use std::path::Path;
use std::sync::RwLock;

use crate::core::index::hnsw::{HnswIndex, SearchResult as HnswSearchResult};

pub struct FaceIndex {
    index_path: String,
    dimension: usize,
    index: RwLock<Option<HnswIndex>>,
}

impl FaceIndex {
    pub fn new(data_dir: &Path) -> Self {
        let index_path = data_dir.join("face_index.bin");
        Self {
            index_path: index_path.to_string_lossy().to_string(),
            dimension: 512,
            index: RwLock::new(None),
        }
    }

    pub fn embedding_dim(&self) -> usize {
        self.dimension
    }

    pub async fn load(&mut self) -> Result<()> {
        let idx = HnswIndex::load(&self.index_path, self.dimension)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        *self.index.write().unwrap() = Some(idx);
        Ok(())
    }

    pub async fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let index = self.index.read().unwrap();
        if let Some(index) = index.as_ref() {
            index.search(query, top_k)
                .into_iter()
                .map(|r| SearchResult { id: r.id, score: r.score })
                .collect()
        } else {
            vec![]
        }
    }

    pub async fn add(&mut self, id: i64, embedding: &[f32]) -> Result<()> {
        let mut index = self.index.write().unwrap();
        if let Some(idx) = index.as_mut() {
            idx.add(embedding.to_vec(), id);
        }
        Ok(())
    }

    pub async fn save(&self) -> Result<()> {
        let index = self.index.read().unwrap();
        if let Some(idx) = index.as_ref() {
            idx.save(&self.index_path)?;
        }
        Ok(())
    }

    pub async fn rebuild(&mut self, vectors: &[(i64, Vec<f32>)]) -> Result<()> {
        let mut idx = HnswIndex::new(self.dimension);
        for (id, embedding) in vectors {
            idx.add(embedding.clone(), *id);
        }
        idx.save(&self.index_path)?;
        *self.index.write().unwrap() = Some(idx);
        Ok(())
    }

    pub fn set_index(&self, idx: HnswIndex) {
        *self.index.write().unwrap() = Some(idx);
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: i64,
    pub score: f32,
}