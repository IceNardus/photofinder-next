//! Recall Test Framework for Patch-based Image Search
//!
//! Tests Recall@K metrics for both coarse (HNSW) and fine (LightGlue) search.
//!
//! # Dataset Structure
//!
//! Test dataset should be organized as:
//!
//! ```text
//! test_dataset/
//! ├── queries/
//! │   ├── query_001.jpg
//! │   └── query_002.jpg
//! └── database/
//!     ├── img_001.jpg
//!     ├── img_002.jpg
//!     └── ...
//!
//! # Ground Truth Format (YAML)
//!
//! ```yaml
//! query_001.jpg:
//!   relevant:
//!     - img_001.jpg
//!     - img_003.jpg
//! query_002.jpg:
//!   relevant:
//!     - img_002.jpg
//! ```
//!
//! # Usage
//!
//! ```rust
//! use recall_test::{RecallEvaluator, RecallConfig};
//!
//! let config = RecallConfig {
//!     dataset_path: "test_dataset".to_string(),
//!     ground_truth_path: "test_dataset/ground_truth.yaml".to_string(),
//!     ..Default::default()
//! };
//!
//! let mut evaluator = RecallEvaluator::new(config)?;
//! let results = evaluator.run_coarse_recall_test(&patch_search, 10)?;
//! evaluator.print_results(&results);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Configuration for recall testing
#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// Path to test dataset directory
    pub dataset_path: String,
    /// Path to ground truth YAML file
    pub ground_truth_path: String,
    /// Path to SuperPoint model
    pub superpoint_path: String,
    /// Path to LightGlue model
    pub lightglue_path: String,
    /// Path to HNSW index file
    pub index_path: String,
    /// Database path
    pub db_path: String,
    /// Maximum keypoints per patch
    pub max_keypoints_per_patch: usize,
    /// HNSW search top_k
    pub hnsw_top_k: usize,
    /// LightGlue minimum matches
    pub min_lightglue_matches: usize,
    /// RANSAC inlier threshold
    pub ransac_threshold: f32,
    /// Early exit: max keypoint difference
    pub max_kpt_difference: usize,
    /// Early exit: min descriptor similarity
    pub min_desc_similarity: f32,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            dataset_path: "tests/recall/data".to_string(),
            ground_truth_path: "tests/recall/ground_truth.yaml".to_string(),
            superpoint_path: "resources/models/superpoint.onnx".to_string(),
            lightglue_path: "resources/models/lightglue.mnn".to_string(),
            index_path: "data/patch_hnsw.idx".to_string(),
            db_path: "data/photofinder.db".to_string(),
            max_keypoints_per_patch: 256,
            hnsw_top_k: 50,
            min_lightglue_matches: 15,
            ransac_threshold: 4.0,
            max_kpt_difference: 200,
            min_desc_similarity: 0.7,
        }
    }
}

/// Ground truth entry for a query
#[derive(Debug, Clone, Deserialize)]
pub struct GroundTruthEntry {
    /// List of relevant image filenames (just filenames, not full paths)
    pub relevant: Vec<String>,
    /// Optional: list of irrelevant images for precision testing
    #[serde(default)]
    pub irrelevant: Vec<String>,
}

/// Ground truth data structure
#[derive(Debug, Clone, Deserialize)]
pub struct GroundTruth(pub HashMap<String, GroundTruthEntry>);

/// A single query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub query_name: String,
    /// List of candidate image names returned by search, ordered by rank
    pub candidates: Vec<String>,
    /// Number of relevant images found in top K
    pub recall_at_k: HashMap<usize, f32>,
    /// Whether the top-1 result is correct
    pub top1_correct: bool,
    /// Whether any relevant image was found
    pub any_relevant_found: bool,
}

/// Aggregated recall metrics
#[derive(Debug, Clone, Serialize)]
pub struct RecallMetrics {
    pub total_queries: usize,
    pub recall_at_1: f32,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub recall_at_20: f32,
    pub mrr: f32,  // Mean Reciprocal Rank
    pub top1_accuracy: f32,
    pub coverage: f32,  // % of queries with at least one relevant result
}

impl RecallMetrics {
    pub fn new() -> Self {
        Self {
            total_queries: 0,
            recall_at_1: 0.0,
            recall_at_5: 0.0,
            recall_at_10: 0.0,
            recall_at_20: 0.0,
            mrr: 0.0,
            top1_accuracy: 0.0,
            coverage: 0.0,
        }
    }

    /// Compute metrics from query results
    pub fn compute(results: &[QueryResult]) -> Self {
        if results.is_empty() {
            return Self::new();
        }

        let total = results.len() as f32;
        let mut metrics = Self::new();
        metrics.total_queries = results.len();

        let mut mrr_sum = 0.0f32;
        let mut top1_correct = 0usize;
        let mut queries_with_relevant = 0usize;

        for r in results {
            // Recall@K
            for k in &[1usize, 5, 10, 20] {
                let recall = r.recall_at_k.get(k).copied().unwrap_or(0.0);
                match k {
                    1 => metrics.recall_at_1 += recall,
                    5 => metrics.recall_at_5 += recall,
                    10 => metrics.recall_at_10 += recall,
                    20 => metrics.recall_at_20 += recall,
                    _ => {}
                }
            }

            // MRR: reciprocal rank of first relevant result
            for (rank, cand) in r.candidates.iter().enumerate() {
                if r.recall_at_k.get(&1).map_or(false, |_| true) {
                    // Check if this candidate is relevant
                    if r.any_relevant_found {
                        mrr_sum += 1.0 / (rank + 1) as f32;
                        break;
                    }
                }
            }

            // Top-1 accuracy
            if r.top1_correct {
                top1_correct += 1;
            }

            // Coverage
            if r.any_relevant_found {
                queries_with_relevant += 1;
            }
        }

        metrics.recall_at_1 /= total;
        metrics.recall_at_5 /= total;
        metrics.recall_at_10 /= total;
        metrics.recall_at_20 /= total;
        metrics.mrr = mrr_sum / total;
        metrics.top1_accuracy = top1_correct as f32 / total;
        metrics.coverage = queries_with_relevant as f32 / total;

        metrics
    }
}

/// Recall test framework
pub struct RecallEvaluator {
    config: RecallConfig,
    ground_truth: GroundTruth,
}

impl RecallEvaluator {
    /// Create a new recall evaluator
    pub fn new(config: RecallConfig) -> Result<Self, String> {
        // Load ground truth
        let gt_path = Path::new(&config.ground_truth_path);
        if !gt_path.exists() {
            return Err(format!("Ground truth file not found: {:?}", gt_path));
        }

        let gt_content = std::fs::read_to_string(gt_path)
            .map_err(|e| format!("Failed to read ground truth: {}", e))?;

        let ground_truth: GroundTruth = serde_yaml::from_str(&gt_content)
            .map_err(|e| format!("Failed to parse ground truth YAML: {}", e))?;

        info!("Loaded ground truth with {} queries", ground_truth.0.len());

        Ok(Self { config, ground_truth })
    }

    /// Get list of query images from the dataset
    pub fn get_query_images(&self) -> Result<Vec<(String, PathBuf)>, String> {
        let queries_dir = Path::new(&self.config.dataset_path).join("queries");
        if !queries_dir.exists() {
            return Err(format!("Queries directory not found: {:?}", queries_dir));
        }

        let mut queries = Vec::new();
        for entry in std::fs::read_dir(&queries_dir)
            .map_err(|e| format!("Failed to read queries dir: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Entry error: {}", e))?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    let name = path.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    queries.push((name, path));
                }
            }
        }

        queries.sort();
        Ok(queries)
    }

    /// Get database images
    pub fn get_database_images(&self) -> Result<Vec<(String, PathBuf)>, String> {
        let db_dir = Path::new(&self.config.dataset_path).join("database");
        if !db_dir.exists() {
            return Err(format!("Database directory not found: {:?}", db_dir));
        }

        let mut images = Vec::new();
        for entry in std::fs::read_dir(&db_dir)
            .map_err(|e| format!("Failed to read database dir: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Entry error: {}", e))?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    let name = path.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    images.push((name, path));
                }
            }
        }

        images.sort();
        Ok(images)
    }

    /// Run coarse recall test (HNSW only)
    ///
    /// Returns per-query results
    pub fn run_coarse_recall_test<S>(
        &mut self,
        search: &S,
        max_k: usize,
    ) -> Result<Vec<QueryResult>, String>
    where
        S: CoarseSearch,
    {
        let queries = self.get_query_images()?;
        let mut results = Vec::new();

        for (query_name, query_path) in queries {
            info!("Processing query: {}", query_name);

            // Get ground truth for this query
            let gt_entry = match self.ground_truth.0.get(&query_name) {
                Some(gt) => gt,
                None => {
                    warn!("No ground truth for query: {}", query_name);
                    continue;
                }
            };

            // Run search
            let candidates = search.search(query_path.to_str().unwrap(), self.config.hnsw_top_k)?;

            // Compute recall metrics
            let result = self.compute_query_result(&query_name, &candidates, gt_entry, max_k);
            results.push(result);
        }

        Ok(results)
    }

    /// Compute result for a single query
    fn compute_query_result(
        &self,
        query_name: &str,
        candidates: &[String],
        gt_entry: &GroundTruthEntry,
        max_k: usize,
    ) -> QueryResult {
        let mut recall_at_k = HashMap::new();
        let relevant_set: std::collections::HashSet<_> = gt_entry.relevant.iter().collect();

        let mut any_relevant_found = false;
        let mut top1_correct = false;

        for k in &[1usize, 5, 10, 20] {
            if *k > max_k {
                continue;
            }

            let top_k_candidates = &candidates[..(*k).min(candidates.len())];
            let num_relevant = top_k_candidates.iter()
                .filter(|c| relevant_set.contains(*c))
                .count();

            recall_at_k.insert(*k, num_relevant as f32 / gt_entry.relevant.len().max(1) as f32);
        }

        // Check top-1 correctness
        if let Some(first) = candidates.first() {
            top1_correct = relevant_set.contains(first);
        }

        // Check if any relevant found
        any_relevant_found = candidates.iter().any(|c| relevant_set.contains(c));

        QueryResult {
            query_name: query_name.to_string(),
            candidates: candidates.to_vec(),
            recall_at_k,
            top1_correct,
            any_relevant_found,
        }
    }

    /// Print aggregated results
    pub fn print_results(&self, results: &[QueryResult]) {
        let metrics = RecallMetrics::compute(results);

        println!();
        println!("========================================");
        println!(" RECALL TEST RESULTS");
        println!("========================================");
        println!("Total queries tested: {}", metrics.total_queries);
        println!("----------------------------------------");
        println!("Recall@1:  {:.1}%", metrics.recall_at_1 * 100.0);
        println!("Recall@5:  {:.1}%", metrics.recall_at_5 * 100.0);
        println!("Recall@10: {:.1}%", metrics.recall_at_10 * 100.0);
        println!("Recall@20: {:.1}%", metrics.recall_at_20 * 100.0);
        println!("----------------------------------------");
        println!("Top-1 Accuracy: {:.1}%", metrics.top1_accuracy * 100.0);
        println!("MRR:            {:.3}", metrics.mrr);
        println!("Coverage:       {:.1}%", metrics.coverage * 100.0);
        println!("========================================");
        println!();

        // Print per-query details for debugging
        println!("Per-query breakdown:");
        println!("----------------------------------------");
        for r in results {
            let r1 = r.recall_at_k.get(&1).copied().unwrap_or(0.0);
            let r5 = r.recall_at_k.get(&5).copied().unwrap_or(0.0);
            let r10 = r.recall_at_k.get(&10).copied().unwrap_or(0.0);
            println!("{}: R@1={:.2} R@5={:.2} R@10={:.2} top1={}",
                     r.query_name, r1, r5, r10,
                     if r.top1_correct { "✓" } else { "✗" });
        }
    }
}

/// Trait for coarse search implementations
pub trait CoarseSearch {
    /// Search and return candidate image names ordered by rank
    fn search(&self, query_path: &str, top_k: usize) -> Result<Vec<String>, String>;
}

/// Fine search result for a single query
#[derive(Debug, Clone)]
pub struct FineQueryResult {
    pub query_name: String,
    pub search_results: Vec<SearchResultEntry>,
}

/// Individual search result entry
#[derive(Debug, Clone)]
pub struct SearchResultEntry {
    pub image_name: String,
    pub total_matches: usize,
    pub inlier_count: usize,
    pub confidence: f32,
}

/// Run fine recall test (LightGlue matching on coarse candidates)
///
/// This tests the full pipeline: HNSW coarse search + LightGlue fine matching
pub struct FineRecallEvaluator {
    config: RecallConfig,
    ground_truth: GroundTruth,
}

impl FineRecallEvaluator {
    pub fn new(config: RecallConfig) -> Result<Self, String> {
        let gt_path = Path::new(&config.ground_truth_path);
        if !gt_path.exists() {
            return Err(format!("Ground truth file not found: {:?}", gt_path));
        }

        let gt_content = std::fs::read_to_string(gt_path)
            .map_err(|e| format!("Failed to read ground truth: {}", e))?;

        let ground_truth: GroundTruth = serde_yaml::from_str(&gt_content)
            .map_err(|e| format!("Failed to parse ground truth YAML: {}", e))?;

        Ok(Self { config, ground_truth })
    }

    pub fn get_query_images(&self) -> Result<Vec<(String, PathBuf)>, String> {
        let queries_dir = Path::new(&self.config.dataset_path).join("queries");
        let mut queries = Vec::new();

        for entry in std::fs::read_dir(&queries_dir)
            .map_err(|e| format!("Failed to read queries dir: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Entry error: {}", e))?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    let name = path.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    queries.push((name, path));
                }
            }
        }

        queries.sort();
        Ok(queries)
    }
}

/// Create a sample ground truth YAML file
pub fn create_sample_ground_truth(path: &Path) -> Result<(), String> {
    let sample = r#"# Sample Ground Truth for Recall Testing
# Format: query_filename:
#   relevant:
#     - database_image_1.jpg
#     - database_image_2.jpg
#   irrelevant:
#     - unrelated_image.jpg

query_001.jpg:
  relevant:
    - img_001.jpg
    - img_002.jpg

query_002.jpg:
  relevant:
    - img_003.jpg
"#;

    std::fs::write(path, sample)
        .map_err(|e| format!("Failed to write sample ground truth: {}", e))?;

    Ok(())
}

/// Create dataset directory structure
pub fn create_dataset_structure(base_path: &Path) -> Result<(), String> {
    let queries = base_path.join("queries");
    let database = base_path.join("database");

    std::fs::create_dir_all(&queries)
        .map_err(|e| format!("Failed to create queries dir: {}", e))?;
    std::fs::create_dir_all(&database)
        .map_err(|e| format!("Failed to create database dir: {}", e))?;

    info!("Created dataset structure at {:?}", base_path);
    info!("  - {:?}", queries);
    info!("  - {:?}", database);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_metrics_computation() {
        let results = vec![
            QueryResult {
                query_name: "q1.jpg".to_string(),
                candidates: vec!["img1.jpg".to_string(), "img2.jpg".to_string()],
                recall_at_k: [(1, 1.0), (5, 1.0)].into(),
                top1_correct: true,
                any_relevant_found: true,
            },
            QueryResult {
                query_name: "q2.jpg".to_string(),
                candidates: vec!["img3.jpg".to_string(), "img1.jpg".to_string()],
                recall_at_k: [(1, 0.0), (5, 1.0)].into(),
                top1_correct: false,
                any_relevant_found: true,
            },
        ];

        let metrics = RecallMetrics::compute(&results);
        assert_eq!(metrics.total_queries, 2);
        assert_eq!(metrics.recall_at_1, 0.5);  // 1 out of 2 correct at @1
        assert_eq!(metrics.recall_at_5, 1.0);   // 100% at @5
        assert_eq!(metrics.top1_accuracy, 0.5);
    }

    #[test]
    fn test_empty_results() {
        let metrics = RecallMetrics::compute(&[]);
        assert_eq!(metrics.total_queries, 0);
    }

    #[test]
    fn test_create_sample_ground_truth() {
        let temp_dir = std::env::temp_dir();
        let gt_path = temp_dir.join("test_ground_truth.yaml");

        create_sample_ground_truth(&gt_path).unwrap();

        let content = std::fs::read_to_string(&gt_path).unwrap();
        let gt: GroundTruth = serde_yaml::from_str(&content).unwrap();
        assert!(gt.0.contains_key("query_001.jpg"));

        std::fs::remove_file(&gt_path).ok();
    }
}