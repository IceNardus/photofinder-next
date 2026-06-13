use anyhow::Result;
use std::path::Path;

pub struct HnswIndex {
    dimension: usize,
    vectors: Vec<Vec<f32>>,
    ids: Vec<i64>,
}

impl HnswIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
            ids: Vec::new(),
        }
    }

    pub fn add(&mut self, vector: Vec<f32>, id: i64) {
        self.vectors.push(vector);
        self.ids.push(id);
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn get_all(&self) -> Vec<(i64, Vec<f32>)> {
        self.ids.iter()
            .zip(self.vectors.iter())
            .map(|(id, v)| (*id, v.clone()))
            .collect()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn vectors_ref(&self) -> &Vec<Vec<f32>> {
        &self.vectors
    }

    pub fn ids_ref(&self) -> &Vec<i64> {
        &self.ids
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
        self.ids.clear();
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if self.vectors.is_empty() {
            return vec![];
        }

        let mut scores: Vec<(usize, f32)> = self.vectors.iter()
            .enumerate()
            .map(|(idx, v)| (idx, cosine_sim(query, v)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scores.into_iter()
            .take(top_k)
            .map(|(idx, score)| SearchResult {
                id: self.ids[idx],
                score,
            })
            .collect()
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut data = Vec::new();
        data.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        data.extend_from_slice(&(self.vectors.len() as u32).to_le_bytes());
        for (id, v) in self.ids.iter().zip(&self.vectors) {
            data.extend_from_slice(&id.to_le_bytes());
            for f in v {
                data.extend_from_slice(&f.to_le_bytes());
            }
        }
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &str, dimension: usize) -> Result<Self, String> {
        let path = Path::new(path);
        if !path.exists() {
            return Ok(Self::new(dimension));
        }
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut offset = 0;
        let dim = u32::from_le_bytes(data[offset..4].try_into().unwrap()) as usize;
        offset += 4;
        let count = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;

        let mut index = Self::new(dim);
        for _ in 0..count {
            let id = i64::from_le_bytes(data[offset..offset+8].try_into().unwrap());
            offset += 8;
            let mut vector = vec![0.0; dim];
            for i in 0..dim {
                vector[i] = f32::from_le_bytes(data[offset..offset+4].try_into().unwrap());
                offset += 4;
            }
            index.add(vector, id);
        }
        Ok(index)
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: i64,
    pub score: f32,
}