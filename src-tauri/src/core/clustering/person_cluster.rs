use std::sync::Arc;
use std::path::Path;
use anyhow::Result;
use tracing::info;

use crate::core::database::Database;
use crate::core::database::faces::FacesTable;
use crate::core::database::persons::PersonsTable;
use crate::core::storage::FaceVectorStore;
use crate::core::index::hnsw::HnswIndex;

const MATCH_THRESHOLD: f32 = 0.65;
const DIMENSION: usize = 512;

pub struct PersonCluster {
    db: Arc<Database>,
    vector_store: Arc<FaceVectorStore>,
    index: HnswIndex,
    index_loaded: bool,
    data_dir: std::path::PathBuf,
}

impl PersonCluster {
    pub fn new(db: Arc<Database>, data_dir: &Path) -> Result<Self> {
        let vector_store = Arc::new(FaceVectorStore::new(data_dir));
        vector_store.init()?;
        let index = HnswIndex::new(DIMENSION);

        Ok(Self {
            db,
            vector_store,
            index,
            index_loaded: false,
            data_dir: data_dir.to_path_buf(),
        })
    }

    fn index_path(&self) -> std::path::PathBuf {
        let path = self.data_dir.join("vectors").join("person_cluster.bin");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        path
    }

    fn auto_save(&self) {
        let path = self.index_path();
        info!("PersonCluster auto_save: saving to {:?}", path);
        if let Err(e) = self.save_index(&path.to_string_lossy()) {
            info!("Failed to auto-save person cluster index: {}", e);
        } else {
            info!("PersonCluster auto_save: success, count={}", self.index.len());
        }
    }

    pub fn load_index(&mut self, index_path: &str) -> Result<()> {
        if std::path::Path::new(index_path).exists() {
            self.index = HnswIndex::load(index_path, DIMENSION)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            info!("Person cluster index loaded from {}, count={}", index_path, self.index.len());
        } else {
            self.index = HnswIndex::new(DIMENSION);
            info!("Person cluster index file not found, created new index at {}", index_path);
        }
        self.index_loaded = true;
        Ok(())
    }

    pub fn save_index(&self, index_path: &str) -> Result<()> {
        self.index.save(index_path)?;
        info!("Person cluster index saved to {}", index_path);
        Ok(())
    }

    pub fn assign_person(&mut self, embedding: &[f32]) -> Result<i64> {
        if !self.index_loaded {
            return Err(anyhow::anyhow!("Index not loaded"));
        }

        // k=10 search for more robust voting
        let k = 10.min(self.index.len().max(1));
        let results = self.index.search(embedding, k);

        if results.is_empty() {
            info!("PersonCluster: No search results");
            return Err(anyhow::anyhow!("No search results"));
        }

        // Count votes by person_id
        let mut person_votes: std::collections::HashMap<i64, (u32, f32)> = std::collections::HashMap::new();
        for result in &results {
            let entry = person_votes.entry(result.id).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += result.score;
        }

        // Find person with most votes, breaking ties by total score
        let mut best_person_id = results[0].id;
        let mut best_votes = 0u32;
        let mut best_total_score = 0.0f32;

        for (person_id, (votes, total_score)) in &person_votes {
            let normalized_score = *total_score / *votes as f32;
            let better = if *votes > best_votes {
                true
            } else if *votes == best_votes && total_score > &best_total_score {
                true
            } else {
                false
            };

            if better {
                best_votes = *votes;
                best_person_id = *person_id;
                best_total_score = normalized_score;
            }
        }

        let top_score = results[0].score;
        info!("PersonCluster assign: k={}, best_person={}, votes={}, top_score={:.3}, avg_score={:.3}",
              k, best_person_id, best_votes, top_score, best_total_score);

        if top_score >= MATCH_THRESHOLD {
            Ok(best_person_id)
        } else {
            Err(anyhow::anyhow!("No matching person found (top_score {:.3} < {:.3})", top_score, MATCH_THRESHOLD))
        }
    }

    pub fn create_person(&mut self, face_id: i64, embedding: &[f32]) -> Result<i64> {
        let person_id = PersonsTable::add(
            &self.db.conn,
            1,
            Some(face_id),
            0,
            DIMENSION as i32,
        )?;

        self.index.add(embedding.to_vec(), person_id);
        self.auto_save();
        FacesTable::update_person_id(&self.db.conn, face_id, person_id)?;

        info!("Created new person {} for face {}", person_id, face_id);
        Ok(person_id)
    }

    pub fn add_face_to_person(&mut self, face_id: i64, person_id: i64, embedding: &[f32]) -> Result<()> {
        PersonsTable::increment_face_count(&self.db.conn, person_id)?;
        self.index.add(embedding.to_vec(), person_id);
        self.auto_save();
        FacesTable::update_person_id(&self.db.conn, face_id, person_id)?;
        info!("Added face {} to person {}", face_id, person_id);
        Ok(())
    }

    pub fn get_person_face_ids(&self, person_id: i64) -> Result<Vec<i64>> {
        let faces = FacesTable::get_by_person(&self.db.conn, person_id)?;
        Ok(faces.into_iter().map(|f| f.id).collect())
    }

    pub fn get_person_count(&self) -> Result<i64> {
        PersonsTable::count(&self.db.conn)
    }

    pub fn rebuild_index(&mut self) -> Result<()> {
        let persons = PersonsTable::get_all(&self.db.conn)?;
        let mut new_index = HnswIndex::new(DIMENSION);

        for person in &persons {
            if person.center_vector_offset > 0 {
                if let Ok(embedding) = self.vector_store.get_vector(person.center_vector_offset, person.center_vector_length) {
                    new_index.add(embedding, person.id);
                }
            }
        }

        self.index = new_index;
        self.index_loaded = true;
        info!("Rebuilt person cluster index with {} persons", persons.len());
        Ok(())
    }
}