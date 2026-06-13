use std::sync::Arc;
use anyhow::Result;
use tracing::info;

use crate::core::database::Database;
use crate::core::storage::FaceVectorStore;
use crate::core::index::FaceIndex;

pub struct PersonPipeline {
    db: Arc<Database>,
    face_store: FaceVectorStore,
    face_index: FaceIndex,
}

impl PersonPipeline {
    pub fn new(db: Arc<Database>, face_store: FaceVectorStore, face_index: FaceIndex) -> Self {
        Self {
            db,
            face_store,
            face_index,
        }
    }

    pub async fn process_image(&self, image_path: &str) -> Result<Vec<PersonResult>> {
        info!("Processing person for image: {}", image_path);
        Ok(vec![])
    }
}

pub struct PersonResult {
    pub embedding_id: String,
    pub quality: f32,
    pub bbox: [f32; 4],
}