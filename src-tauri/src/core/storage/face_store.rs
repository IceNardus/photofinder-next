use anyhow::Result;
use std::path::{Path, PathBuf};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::sync::RwLock;
use tracing::info;
use serde::{Serialize, Deserialize};

const VECTOR_FILE_MAGIC: &[u8; 8] = b"FVECTBIN";
const CURRENT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFileHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub dimension: u32,
    pub model_name: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: i64,
    offset: u64,
    length: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct FaceVectorLocation {
    pub offset: u64,
    pub length: u32,
    pub vector_length: i32,
}

pub struct FaceVectorStore {
    index_path: PathBuf,
    meta_path: PathBuf,
    header: RwLock<VectorFileHeader>,
    entries: RwLock<Vec<VectorEntry>>,
}

impl FaceVectorStore {
    pub fn index_path(&self) -> &Path {
        &self.index_path
    }
    pub fn new(data_dir: &Path) -> Self {
        let index_path = data_dir.join("vectors").join("face_vectors.bin");
        let meta_path = data_dir.join("vectors").join("face_vectors.meta");
        Self {
            index_path,
            meta_path,
            header: RwLock::new(VectorFileHeader {
                magic: *VECTOR_FILE_MAGIC,
                version: CURRENT_VERSION,
                dimension: 512,
                model_name: "ArcFace_w600k_r50".to_string(),
                count: 0,
            }),
            entries: RwLock::new(Vec::new()),
        }
    }

    pub fn init(&self) -> Result<()> {
        info!("[FACE_STORE] init: starting, index_path={}", self.index_path.display());
        info!("[FACE_STORE] init: index_path.exists()={}", self.index_path.exists());
        info!("[FACE_STORE] init: meta_path={}", self.meta_path.display());
        if let Some(parent) = self.index_path.parent() {
            std::fs::create_dir_all(parent)?;
            info!("[FACE_STORE] init: created parent dir if needed");
        }

        // Only create new if index file doesn't exist
        if !self.index_path.exists() {
            info!("[FACE_STORE] init: creating new file");
            self.create_new()?;
        } else {
            // Index file exists, try to load it
            info!("[FACE_STORE] init: loading existing");
            match self.load_header() {
                Ok(_) => {
                    info!("[FACE_STORE] init: header loaded successfully");
                    // Load entries if meta exists
                    if self.meta_path.exists() {
                        info!("[FACE_STORE] init: meta exists, loading entries");
                        self.load_entries()?;
                    } else {
                        info!("[FACE_STORE] init: meta does NOT exist");
                    }
                }
                Err(e) => {
                    // Header invalid - recreate the file
                    info!("[FACE_STORE] init: header load failed ({}), recreating", e);
                    // Backup old file first
                    let backup_path = self.index_path.with_extension("bin.old");
                    if let Err(e2) = std::fs::rename(&self.index_path, &backup_path) {
                        info!("[FACE_STORE] init: backup failed: {}, deleting old file", e2);
                        let _ = std::fs::remove_file(&self.index_path);
                    }
                    self.create_new()?;
                }
            }
        }

        info!("[FACE_STORE] init: {} entries, meta_path={}", self.entries.read().unwrap().len(), self.meta_path.display());
        Ok(())
    }

    fn create_new(&self) -> Result<()> {
        File::create(&self.index_path)?;
        *self.entries.write().unwrap() = Vec::new();
        self.save_header()?;
        Ok(())
    }

    fn load_header(&self) -> Result<()> {
        let mut file = File::open(&self.index_path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;

        if &magic != VECTOR_FILE_MAGIC {
            return Err(anyhow::anyhow!("Invalid vector file format"));
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes);

        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;

        let model_name = Self::read_string(&mut file, 64)?;

        *self.header.write().unwrap() = VectorFileHeader {
            magic,
            version,
            dimension,
            model_name,
            count,
        };

        Ok(())
    }

    fn load_entries(&self) -> Result<()> {
        // Load entries from meta file regardless of header count
        // (header count might be 0 if we haven't saved after adding new entries)
        eprintln!("[STORE] load_entries: meta_path={}", self.meta_path.display());
        eprintln!("[STORE] load_entries: meta_path.exists()={}", self.meta_path.exists());

        if !self.meta_path.exists() {
            return Ok(());
        }

        let data = std::fs::read(&self.meta_path)?;
        eprintln!("[STORE] load_entries: read {} bytes from meta", data.len());
        let entries: Vec<VectorEntry> = serde_json::from_slice(&data)?;
        let count = entries.len();
        eprintln!("[STORE] load_entries: loaded {} entries from meta", count);
        *self.entries.write().unwrap() = entries;

        // Update header count to match
        self.header.write().unwrap().count = count;

        Ok(())
    }

    fn save_header(&self) -> Result<()> {
        let header = self.header.read().unwrap();

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(&self.index_path)?;
        file.seek(std::io::SeekFrom::Start(0))?;
        file.write_all(VECTOR_FILE_MAGIC)?;
        file.write_all(&header.version.to_le_bytes())?;
        file.write_all(&header.dimension.to_le_bytes())?;
        file.write_all(&(header.count as u64).to_le_bytes())?;
        let model_name = Self::pad_string(&header.model_name, 64);
        file.write_all(model_name.as_bytes())?;
        Ok(())
    }

    pub fn save(&self, id: i64, embedding: &[f32]) -> Result<FaceVectorLocation> {
        info!("[FACE_STORE] save: id={}, embedding_len={}", id, embedding.len());
        info!("[FACE_STORE] save: index_path={}", self.index_path.display());
        info!("[FACE_STORE] save: index_path.exists()={}", self.index_path.exists());

        // If file doesn't exist, recreate it
        if !self.index_path.exists() {
            info!("[FACE_STORE] save: file missing, recreating");
            self.create_new()?;
        }

        let offset = std::fs::metadata(&self.index_path)?.len();
        info!("[FACE_STORE] save: current file size (offset)={}", offset);
        let length = (embedding.len() * 4) as u32;

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(&self.index_path)?;
        let data: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&data)?;

        let mut entries = self.entries.write().unwrap();
        entries.push(VectorEntry { id, offset, length });
        self.header.write().unwrap().count = entries.len();

        drop(entries);
        self.save_checkpoint()?;

        // Update bin header count - need to close and re-open without append mode
        let count = self.header.read().unwrap().count;
        drop(self.header.read().unwrap());

        let mut file = OpenOptions::new()
            .write(true)
            .open(&self.index_path)?;
        file.seek(std::io::SeekFrom::Start(16))?;
        file.write_all(&(count as u64).to_le_bytes())?;

        Ok(FaceVectorLocation { offset, length, vector_length: length as i32 })
    }

    fn save_checkpoint(&self) -> Result<()> {
        let entries = self.entries.read().unwrap();
        info!("[FACE_STORE] save_checkpoint: writing {} entries to meta_path={}", entries.len(), self.meta_path.display());
        let data = serde_json::to_vec(&*entries)?;
        if let Err(e) = std::fs::write(&self.meta_path, data) {
            info!("[FACE_STORE] save_checkpoint: FAILED to write meta: {}", e);
            return Err(anyhow::anyhow!("Failed to write meta: {}", e));
        }
        info!("[FACE_STORE] save_checkpoint: succeeded");
        Ok(())
    }

    pub fn load(&self, offset: u64, length: u32) -> Result<Vec<f32>> {
        let mut file = File::open(&self.index_path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut data = vec![0u8; length as usize];
        file.read_exact(&mut data)?;
        let embedding = data.chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(embedding)
    }

    pub fn get_vector(&self, offset: i64, length: i32) -> Result<Vec<f32>> {
        self.load(offset as u64, length as u32)
    }

    pub fn delete_vector(&self, offset: u64) -> Result<()> {
        let mut entries = self.entries.write().unwrap();
        entries.retain(|e| e.offset != offset);
        drop(entries);
        self.save_checkpoint()?;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.header.read().unwrap().count
    }

    pub fn get_all_vectors(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        // Reload entries from disk first to get the latest state
        // This is needed because init() might have been called before vectors were saved
        if self.meta_path.exists() {
            let data = std::fs::read(&self.meta_path)?;
            let new_entries: Vec<VectorEntry> = serde_json::from_slice(&data)?;
            let count = new_entries.len();
            *self.entries.write().unwrap() = new_entries;
            self.header.write().unwrap().count = count;
        }

        let entries = self.entries.read().unwrap();
        info!("[FACE_STORE] get_all_vectors: {} entries", entries.len());
        let mut results = Vec::with_capacity(entries.len());

        // Debug: track previous embedding to detect duplicates
        let mut prev_id: Option<i64> = None;
        let mut prev_sum: Option<f32> = None;
        let mut prev_first5: Option<String> = None;

        for entry in entries.iter() {
            let embedding = self.load(entry.offset, entry.length)?;

            let emb_sum: f32 = embedding.iter().sum();
            let emb_first5: String = embedding.iter().take(5)
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(",");

            // Check for duplicate embeddings
            if let (Some(pid), Some(psum), Some(pf5)) = (prev_id, prev_sum, prev_first5.clone()) {
                let sum_diff = (emb_sum - psum).abs();
                if sum_diff < 0.01 {
                    eprintln!("[FACE_STORE] WARNING: face_id {} sum={:.4} first5=[{}] - PREVIOUS face_id {} sum={:.4} first5=[{}] ALMOST IDENTICAL!",
                        entry.id, emb_sum, emb_first5, pid, psum, pf5);
                }
            }

            prev_id = Some(entry.id);
            prev_sum = Some(emb_sum);
            prev_first5 = Some(emb_first5);

            results.push((entry.id, embedding));
        }
        Ok(results)
    }

    pub fn update_id(&self, offset: u64, new_id: i64) -> Result<()> {
        let mut entries = self.entries.write().unwrap();
        for entry in entries.iter_mut() {
            if entry.offset == offset {
                entry.id = new_id;
                break;
            }
        }
        drop(entries);
        self.save_checkpoint()
    }

    pub fn force_checkpoint(&self) -> Result<()> {
        self.save_checkpoint()
    }

    fn read_string(file: &mut File, len: usize) -> Result<String> {
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)?;
        let s = String::from_utf8(buf)?;
        Ok(s.trim_end_matches('\0').to_string())
    }

    fn pad_string(s: &str, len: usize) -> String {
        let bytes = s.as_bytes();
        let mut result = vec![0u8; len];
        result[..bytes.len()].copy_from_slice(bytes);
        String::from_utf8(result).map_err(|_| anyhow::anyhow!("Invalid string")).unwrap()
    }
}