//! Object Vector Store - Binary vector storage for object embeddings
//!
//! File format:
//! object_vectors.bin - raw f32 vectors, 512 dimensions per vector
//! object_vectors.meta - metadata (object_id, offset)

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use anyhow::{Result, anyhow};
use tracing::info;

const VECTOR_DIM: usize = 512;
const VECTOR_SIZE: usize = VECTOR_DIM * 4; // 2048 bytes per vector

#[derive(Debug, Clone)]
pub struct VectorMeta {
    pub object_id: i64,
    pub offset: u64,
    pub image_id: i64,
    pub image_path: String,
    pub bbox: (f32, f32, f32, f32),  // x1, y1, x2, y2
    pub region_type: String,
}

pub struct VectorStore {
    data_path: std::path::PathBuf,
    meta_path: std::path::PathBuf,
    index_path: std::path::PathBuf,
    count: usize,
}

impl VectorStore {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let data_path = data_dir.join("vectors").join("object_vectors.bin");
        let meta_path = data_dir.join("vectors").join("object_vectors.meta");
        let index_path = data_dir.join("vectors").join("object_index.bin");

        // Create vectors directory
        std::fs::create_dir_all(data_dir.join("vectors"))?;

        let count = if data_path.exists() {
            let metadata = std::fs::metadata(&data_path)?;
            (metadata.len() / VECTOR_SIZE as u64) as usize
        } else {
            0
        };

        Ok(Self {
            data_path,
            meta_path,
            index_path,
            count,
        })
    }

    /// Add a vector and return its offset
    pub fn add(&mut self, object_id: i64, image_id: i64, image_path: &str, bbox: (f32, f32, f32, f32), region_type: &str, embedding: &[f32]) -> Result<u64> {
        if embedding.len() != VECTOR_DIM {
            return Err(anyhow!("Invalid embedding dimension: {}", embedding.len()));
        }

        let offset = self.count as u64 * VECTOR_SIZE as u64;

        // Append vector to binary file
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.data_path)?;

        // Write f32 bytes directly
        let bytes: Vec<u8> = embedding.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        file.write_all(&bytes)?;
        drop(file);

        // Append metadata
        let meta = VectorMeta {
            object_id,
            offset,
            image_id,
            image_path: image_path.to_string(),
            bbox,
            region_type: region_type.to_string(),
        };
        self.append_meta(&meta)?;

        self.count += 1;
        Ok(offset)
    }

    fn append_meta(&self, meta: &VectorMeta) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.meta_path)?;

        // Format: object_id(8) + offset(8) + image_id(8) + path_len(4) + path + bbox(16) + region_type_len(4) + region_type
        let path_bytes = meta.image_path.as_bytes();
        let region_bytes = meta.region_type.as_bytes();

        file.write_all(&meta.object_id.to_le_bytes())?;
        file.write_all(&meta.offset.to_le_bytes())?;
        file.write_all(&meta.image_id.to_le_bytes())?;
        file.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
        file.write_all(path_bytes)?;
        file.write_all(&meta.bbox.0.to_le_bytes())?;
        file.write_all(&meta.bbox.1.to_le_bytes())?;
        file.write_all(&meta.bbox.2.to_le_bytes())?;
        file.write_all(&meta.bbox.3.to_le_bytes())?;
        file.write_all(&(region_bytes.len() as u32).to_le_bytes())?;
        file.write_all(region_bytes)?;
        drop(file);

        Ok(())
    }

    /// Get vector by object_id
    pub fn get(&self, object_id: i64) -> Result<Option<(Vec<f32>, VectorMeta)>> {
        let metas = self.load_metas()?;
        for meta in metas {
            if meta.object_id == object_id {
                let mut file = File::open(&self.data_path)?;
                file.seek(SeekFrom::Start(meta.offset))?;
                let mut bytes = vec![0u8; VECTOR_SIZE];
                file.read_exact(&mut bytes)?;
                let embedding: Vec<f32> = bytes.chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                return Ok(Some((embedding, meta)));
            }
        }
        Ok(None)
    }

    /// Load all metadata
    pub fn load_metas(&self) -> Result<Vec<VectorMeta>> {
        if !self.meta_path.exists() {
            return Ok(vec![]);
        }

        let mut file = File::open(&self.meta_path)?;
        let mut metas = vec![];
        let mut _buffer = vec![0u8; 1024];

        loop {
            // Read fixed-size header: object_id(8) + offset(8) + image_id(8) + path_len(4) = 28 bytes
            let header_size = 28;
            let mut header = vec![0u8; header_size];
            match file.read(&mut header) {
                Ok(0) => break, // EOF
                Ok(n) if n < header_size => break, // Truncated
                Err(e) => return Err(anyhow!("Failed to read meta header: {}", e)),
                Ok(_) => {}
            }

            let object_id = i64::from_le_bytes([header[0], header[1], header[2], header[3], header[4], header[5], header[6], header[7]]);
            let offset = u64::from_le_bytes([header[8], header[9], header[10], header[11], header[12], header[13], header[14], header[15]]);
            let image_id = i64::from_le_bytes([header[16], header[17], header[18], header[19], header[20], header[21], header[22], header[23]]);
            let path_len = u32::from_le_bytes([header[24], header[25], header[26], header[27]]) as usize;

            // Read path
            let mut path_bytes = vec![0u8; path_len];
            file.read_exact(&mut path_bytes)?;
            let image_path = String::from_utf8(path_bytes)?;

            // Read bbox (16 bytes: 4 x f32)
            let mut bbox_bytes = [0u8; 16];
            file.read_exact(&mut bbox_bytes)?;
            let bbox = (
                f32::from_le_bytes([bbox_bytes[0], bbox_bytes[1], bbox_bytes[2], bbox_bytes[3]]),
                f32::from_le_bytes([bbox_bytes[4], bbox_bytes[5], bbox_bytes[6], bbox_bytes[7]]),
                f32::from_le_bytes([bbox_bytes[8], bbox_bytes[9], bbox_bytes[10], bbox_bytes[11]]),
                f32::from_le_bytes([bbox_bytes[12], bbox_bytes[13], bbox_bytes[14], bbox_bytes[15]]),
            );

            // Read region_type_len(4) + region_type
            let mut region_len_bytes = [0u8; 4];
            file.read_exact(&mut region_len_bytes)?;
            let region_len = u32::from_le_bytes(region_len_bytes) as usize;
            let mut region_bytes = vec![0u8; region_len];
            file.read_exact(&mut region_bytes)?;
            let region_type = String::from_utf8(region_bytes)?;

            metas.push(VectorMeta {
                object_id,
                offset,
                image_id,
                image_path,
                bbox,
                region_type,
            });
        }

        Ok(metas)
    }

    /// Load all vectors and metadata for HNSW building
    pub fn load_all(&self) -> Result<(Vec<Vec<f32>>, Vec<VectorMeta>)> {
        if !self.data_path.exists() || !self.meta_path.exists() {
            return Ok((vec![], vec![]));
        }

        let mut file = File::open(&self.data_path)?;
        let metadata = std::fs::metadata(&self.data_path)?;
        let vector_count = (metadata.len() / VECTOR_SIZE as u64) as usize;

        let mut embeddings = Vec::with_capacity(vector_count);
        for i in 0..vector_count {
            let mut bytes = vec![0u8; VECTOR_SIZE];
            file.seek(SeekFrom::Start((i * VECTOR_SIZE) as u64))?;
            file.read_exact(&mut bytes)?;
            let embedding: Vec<f32> = bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            embeddings.push(embedding);
        }

        // Load metas from SQLite instead (simpler)
        let metas = self.load_metas()?;

        Ok((embeddings, metas))
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn clear(&mut self) -> Result<()> {
        if self.data_path.exists() {
            std::fs::remove_file(&self.data_path)?;
        }
        if self.meta_path.exists() {
            std::fs::remove_file(&self.meta_path)?;
        }
        self.count = 0;
        Ok(())
    }
}