use anyhow::Result;
use std::sync::Arc;
use std::path::Path;
use tokio::sync::RwLock;
use tracing::info;

use crate::core::database::Database;
use crate::core::database::images::ImagesTable;
use crate::core::database::faces::FacesTable;
use crate::core::storage::FaceVectorStore;
use crate::core::index::FaceIndex;
use crate::ai::face::{FaceDetector, FaceAligner, ArcFace, DetectedFace};
use crate::core::index::hnsw::HnswIndex;

pub struct PersonSearch {
    db: Arc<Database>,
    face_store: Arc<FaceVectorStore>,
    face_index: Arc<RwLock<FaceIndex>>,
    detector: FaceDetector,
    aligner: FaceAligner,
    arcface: ArcFace,
    data_dir: std::path::PathBuf,
}

impl PersonSearch {
    fn find_models_dir(data_dir: &Path) -> std::path::PathBuf {
        let candidates = vec![
            data_dir.join("resources").join("models"),
            std::path::PathBuf::from("resources/models"),
            std::path::PathBuf::from("/Applications/PhotoFinder Next.app/Contents/Resources/models"),
            std::path::PathBuf::from("/Users/mac/Library/Caches/PhotoFinder/models"),
            std::path::PathBuf::from("/Users/mac/ai-project/photofinder-ai/src-tauri/resources/models"),
        ];

        for candidate in &candidates {
            if candidate.exists() && candidate.join("scrfd_500m_bnkps.onnx").exists() {
                return candidate.clone();
            }
        }

        data_dir.join("resources").join("models")
    }

    pub fn new(db: Arc<Database>, data_dir: &Path) -> Result<Self> {
        let models_dir = Self::find_models_dir(data_dir);

        let scrfd_path = models_dir.join("scrfd_500m_bnkps.onnx");
        let arcface_path = models_dir.join("w600k_r50.onnx");

        let detector = FaceDetector::new(&scrfd_path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("Failed to create detector: {}", e))?;
        let aligner = FaceAligner::new();
        let arcface = ArcFace::new(&arcface_path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("Failed to create arcface: {}", e))?;

        let face_store = Arc::new(FaceVectorStore::new(data_dir));
        face_store.init().map_err(|e| anyhow::anyhow!("face_store.init failed: {}", e))?;
        let count = face_store.count();
        info!("[PERSON_SEARCH] face_store initialized, count={}", count);
        if count == 0 {
            info!("[PERSON_SEARCH] WARNING: face_store has 0 entries!");
        }
        let face_index = Arc::new(RwLock::new(FaceIndex::new(data_dir)));

        Ok(Self {
            db,
            face_store,
            face_index,
            detector,
            aligner,
            arcface,
            data_dir: data_dir.to_path_buf(),
        })
    }

    pub async fn search(&self, query_image_path: &str, _top_k: usize) -> Result<Vec<PersonSearchResult>> {
        eprintln!("[PERSON_SEARCH] ============== START SEARCH ==============");
        eprintln!("[PERSON_SEARCH] Input path: {}", query_image_path);

        // Step 1: Resolve path
        let resolved_path = if std::path::Path::new(query_image_path).is_absolute() {
            query_image_path.to_string()
        } else if query_image_path.starts_with("PhotoFinderNext/") {
            if let Some(base) = dirs::data_local_dir() {
                let filename = query_image_path.strip_prefix("PhotoFinderNext/").unwrap_or(query_image_path);
                base.join("PhotoFinderNext").join(filename).to_string_lossy().to_string()
            } else {
                query_image_path.to_string()
            }
        } else if let Some(base) = dirs::data_local_dir() {
            base.join(query_image_path).to_string_lossy().to_string()
        } else {
            query_image_path.to_string()
        };

        eprintln!("[PERSON_SEARCH] Step 1: Resolved path = {}", resolved_path);
        eprintln!("[PERSON_SEARCH] Step 1: File exists = {}", std::path::Path::new(&resolved_path).exists());

        if !std::path::Path::new(&resolved_path).exists() {
            eprintln!("[PERSON_SEARCH] Step 1: ERROR - File not found!");
            return Err(anyhow::anyhow!("File not found: {}", resolved_path));
        }

        // Step 2: Detect face in query image
        eprintln!("[PERSON_SEARCH] Step 2: Detecting faces...");
        let detected_faces = self.detector.detect(&resolved_path)
            .map_err(|e| anyhow::anyhow!("Detection failed: {}", e))?;

        eprintln!("[PERSON_SEARCH] Step 2: Detected {} faces", detected_faces.len());
        if detected_faces.is_empty() {
            eprintln!("[PERSON_SEARCH] Step 2: ERROR - No faces detected!");
            return Ok(vec![]);
        }

        // List all detected faces with quality scores
        let faces_with_quality: Vec<(usize, &DetectedFace, f32, f32, f32, f32)> = detected_faces
            .iter()
            .enumerate()
            .map(|(i, face)| {
                let face_w = face.bbox[2] - face.bbox[0];
                let face_h = face.bbox[3] - face.bbox[1];
                let min_face_size = face_w.min(face_h);
                let face_area_score = (min_face_size / 150.0).clamp(0.0, 1.0);
                let eye_dist = ((face.keypoints.right_eye.0 - face.keypoints.left_eye.0).powi(2) +
                              (face.keypoints.right_eye.1 - face.keypoints.left_eye.1).powi(2)).sqrt();
                let pose_score = 0.5; // simplified
                let quality = face.score * 0.7 + face_area_score * 0.3;
                (i, face, face.score, face_area_score, eye_dist, pose_score)
            })
            .collect();

        // List all detected faces
        for (i, face, det_score, area_score, eye_dist, pose) in &faces_with_quality {
            let area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]);
            eprintln!("[PERSON_SEARCH] Step 2: Face {} bbox=[{:.0},{:.0},{:.0},{:.0}] area={:.0} det_score={:.3} area_score={:.3} eye_dist={:.1} pose={:.3}",
                     i, face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3], area, det_score, area_score, eye_dist, pose);
        }

        // KPS Validation: check if keypoints fall inside bbox (same as FacePipeline)
        let faces_with_kps_check: Vec<(usize, &DetectedFace, f32, f32, f32, f32, bool)> = faces_with_quality
            .iter()
            .map(|(i, face, det_score, area_score, eye_dist, pose)| {
                let kps = &face.keypoints;
                let kps_in_bbox = kps.left_eye.0 >= face.bbox[0] - 50.0
                    && kps.left_eye.0 <= face.bbox[2] + 50.0
                    && kps.left_eye.1 >= face.bbox[1] - 50.0
                    && kps.left_eye.1 <= face.bbox[3] + 50.0
                    && kps.right_eye.0 >= face.bbox[0] - 50.0
                    && kps.right_eye.0 <= face.bbox[2] + 50.0
                    && kps.right_eye.1 >= face.bbox[1] - 50.0
                    && kps.right_eye.1 <= face.bbox[3] + 50.0;
                eprintln!("[PERSON_SEARCH] Step 2: Face {} KPS_in_bbox={}", i, kps_in_bbox);
                (*i, *face, *det_score, *area_score, *eye_dist, *pose, kps_in_bbox)
            })
            .collect();

        // Apply same quality filters as FacePipeline during scanning (relaxed thresholds)
        // Also apply KPS validation: if keypoints outside bbox, increase threshold
        let filtered_indices: Vec<usize> = faces_with_kps_check
            .iter()
            .filter(|(i, _, det_score, area_score, eye_dist, pose, kps_in_bbox)| {
                // Updated thresholds to match pipeline.rs
                // If KPS outside bbox, require higher detector score
                let min_det_score = if *kps_in_bbox { 0.3 } else { 0.45 };
                let passed = *det_score >= min_det_score && *area_score >= 0.15 && *eye_dist >= 12.0 && *pose >= 0.2;
                if !passed {
                    eprintln!("[PERSON_SEARCH] Step 2: Face {} filtered (det={:.3} area={:.3} eye={:.1} pose={:.3} kps_ok={})",
                             i, det_score, area_score, eye_dist, pose, kps_in_bbox);
                }
                passed
            })
            .map(|(i, _, _, _, _, _, _)| *i)
            .collect();

        eprintln!("[PERSON_SEARCH] Step 2: {} faces passed quality filter", filtered_indices.len());

        // Select the largest face that passed quality filter
        let mut max_area = 0.0f32;
        let mut query_face_idx = filtered_indices[0];
        for &i in &filtered_indices {
            let area = (detected_faces[i].bbox[2] - detected_faces[i].bbox[0]) *
                      (detected_faces[i].bbox[3] - detected_faces[i].bbox[1]);
            if area > max_area {
                max_area = area;
                query_face_idx = i;
            }
        }
        eprintln!("[PERSON_SEARCH] Step 2: Selected face {} for query (area={:.0})", query_face_idx, max_area);

        let query_face = &detected_faces[query_face_idx];
        eprintln!("[PERSON_SEARCH] Step 2: Final selected face idx={}", query_face_idx);

        // Step 3: Align face
        eprintln!("[PERSON_SEARCH] Step 3: Aligning face...");
        let aligned = self.aligner.align(&resolved_path, &query_face.keypoints)
            .map_err(|e| anyhow::anyhow!("Alignment failed: {}", e))?;
        eprintln!("[PERSON_SEARCH] Step 3: Aligned face size: {}x{}", aligned.width(), aligned.height());

        // Step 4: Extract embedding
        eprintln!("[PERSON_SEARCH] Step 4: Extracting embedding...");
        let mut query_embedding = self.arcface.extract(&aligned)
            .map_err(|e| anyhow::anyhow!("Extraction failed: {}", e))?;

        // L2 normalize query embedding
        let norm = query_embedding.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for v in query_embedding.iter_mut() {
            *v /= norm;
        }

        // Debug: check query embedding values
        let query_sum: f32 = query_embedding.iter().sum();
        let query_first5: String = query_embedding.iter().take(5)
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("[PERSON_SEARCH] Step 4: Query embedding dim={} sum={:.4} first5=[{}]", query_embedding.len(), query_sum, query_first5);

        // Step 5: Load vectors from store
        eprintln!("[PERSON_SEARCH] Step 5: Loading vectors from store...");
        let vectors = self.face_store.get_all_vectors()?;
        eprintln!("[PERSON_SEARCH] Step 5: Loaded {} vectors", vectors.len());
        if vectors.is_empty() {
            eprintln!("[PERSON_SEARCH] Step 5: ERROR - No vectors in store!");
            return Ok(vec![]);
        }

        // Debug: check stored vectors
        eprintln!("[PERSON_SEARCH] Step 5: Checking stored vectors...");
        for (i, (face_id, embedding)) in vectors.iter().take(3).enumerate() {
            let emb_sum: f32 = embedding.iter().sum();
            let emb_first5: String = embedding.iter().take(5)
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("[PERSON_SEARCH] Step 5: Stored[{}] face_id={} sum={:.4} first5=[{}]", i, face_id, emb_sum, emb_first5);
        }

        // Step 6: Calculate similarities
        eprintln!("[PERSON_SEARCH] Step 6: Calculating similarities...");
        const SIMILARITY_THRESHOLD: f32 = 0.50;  // Tightened from 0.3 to 0.5 for better precision
        eprintln!("[PERSON_SEARCH] Step 6: Threshold = {}", SIMILARITY_THRESHOLD);

        let mut matches: Vec<(i64, f32)> = Vec::new();
        let mut best_score = 0.0f32;
        let mut best_match_id = 0i64;
        let mut self_score = 0.0f32;
        let mut all_scores: Vec<(i64, f32)> = Vec::new();

        for (face_id, embedding) in &vectors {
            if let None = FacesTable::get_by_id(&self.db.conn, *face_id)? {
                eprintln!("[PERSON_SEARCH] Step 6: face_id={} NOT FOUND in faces table, skipping", face_id);
                continue;
            }
            let score = cosine_sim(&query_embedding, embedding);
            all_scores.push((*face_id, score));

            // Debug: log all similarity scores
            eprintln!("[PERSON_SEARCH] face_id={} similarity={:.4}", face_id, score);

            if *face_id == 2372 {
                self_score = score;
                eprintln!("[PERSON_SEARCH] Step 6: *** SELF TEST face_id=2372 score={:.6} ***", score);
            }

            if score > best_score {
                best_score = score;
                best_match_id = *face_id;
            }
            if score >= SIMILARITY_THRESHOLD {
                matches.push((*face_id, score));
            }
        }

        // Sort all scores for logging
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[PERSON_SEARCH] Step 6: All scores (sorted):");
        for (i, (fid, sc)) in all_scores.iter().take(10).enumerate() {
            eprintln!("[PERSON_SEARCH] Step 6:   {}. face_id={} score={:.4}", i+1, fid, sc);
        }

        eprintln!("[PERSON_SEARCH] Step 6: Best match: face_id={} score={:.4}", best_match_id, best_score);
        eprintln!("[PERSON_SEARCH] Step 6: Matches >= threshold: {}", matches.len());
        eprintln!("[PERSON_SEARCH] ============== END SEARCH ==============");
        info!("[PERSON_SEARCH] Best match: face_id={}, score={:.4}", best_match_id, best_score);
        info!("[PERSON_SEARCH] face_id=2372 self-score={:.6} (threshold={:.1})", self_score, SIMILARITY_THRESHOLD);

        // Sort by similarity descending
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        info!("[PERSON_SEARCH] Found {} matches with similarity >= {}", matches.len(), SIMILARITY_THRESHOLD);

        // Step 4: Get image_ids from face_ids (deduplicated)
        let mut results_map: std::collections::HashMap<i64, PersonSearchResult> = std::collections::HashMap::new();
        for (face_id, score) in &matches {
            if let Some(face_record) = FacesTable::get_by_id(&self.db.conn, *face_id)? {
                // Only add if this image_id not already added, or if this face has higher score
                let entry = results_map.entry(face_record.image_id).or_insert_with(|| PersonSearchResult {
                    image_id: face_record.image_id,
                    face_id: *face_id,
                    thumbnail_path: String::new(),
                    score: *score,
                    bbox: None,
                });

                if entry.score < *score {
                    entry.face_id = *face_id;
                    entry.score = *score;
                }
            }
        }

        // Step 5: Get image paths from image_ids
        let mut results: Vec<PersonSearchResult> = Vec::new();
        for (_, mut result) in results_map {
            if let Some(image) = ImagesTable::get_by_id(&self.db.conn, result.image_id)? {
                info!("[PERSON_SEARCH] Mapping result: image_id={}, path={}", result.image_id, image.path);
                result.thumbnail_path = image.path.clone();
                results.push(result);
            } else {
                info!("[PERSON_SEARCH] No image found for id={}", result.image_id);
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Debug: log all result scores
        eprintln!("[PERSON_SEARCH] Final results ({} items):", results.len());
        for (i, r) in results.iter().take(10).enumerate() {
            eprintln!("  [{}] image_id={} face_id={} score={:.4}", i, r.image_id, r.face_id, r.score);
        }

        info!("[PERSON_SEARCH] Returning {} unique images", results.len());
        Ok(results)
    }

    pub async fn rebuild_index(&self) -> Result<()> {
        let vectors = self.face_store.get_all_vectors()?;

        let mut index = HnswIndex::new(512);
        for (id, embedding) in vectors {
            index.add(embedding, id);
        }

        let index_path = self.data_dir.join("vectors").join("face_index.bin");
        index.save(&index_path.to_string_lossy())?;

        let guard = self.face_index.read().await;
        guard.set_index(index);

        Ok(())
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    // L2 normalize both vectors before computing cosine similarity
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x / norm_a) * (y / norm_b)).sum();

    dot.clamp(-1.0, 1.0)
}

#[derive(Debug, Clone)]
pub struct PersonSearchResult {
    pub image_id: i64,
    pub face_id: i64,
    pub thumbnail_path: String,
    pub score: f32,
    pub bbox: Option<[f32; 4]>,
}