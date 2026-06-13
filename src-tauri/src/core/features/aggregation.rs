//! Descriptor aggregation using VLAD (Vector of Locally Aggregated Descriptors)
//!
//! Aggregates Nx256 SuperPoint descriptors into a single 256-dimensional vector
//! for efficient HNSW indexing.

use super::{DescriptorStats, PatchConfig};

/// VLAD aggregation state
pub struct VladAggregation {
    /// Number of centroids (k-means k)
    k: usize,
    /// Centroids [K, 256]
    centroids: Vec<f32>,
    /// Assignment residuals accumulated
    residuals: Vec<f32>,
    /// Number of descriptors aggregated
    count: usize,
}

impl VladAggregation {
    pub fn new(k: usize) -> Self {
        let centroids = vec![0.0; k * 256];
        let residuals = vec![0.0; k * 256];

        Self {
            k,
            centroids,
            residuals,
            count: 0,
        }
    }

    /// Initialize with pre-computed centroids
    pub fn with_centroids(centroids: Vec<f32>, k: usize) -> Self {
        let residuals = vec![0.0; k * 256];
        Self {
            k,
            centroids,
            residuals,
            count: 0,
        }
    }

    /// Add a descriptor to the VLAD accumulation
    pub fn add_descriptor(&mut self, descriptor: &[f32]) {
        debug_assert_eq!(descriptor.len(), 256);

        // Find nearest centroid
        let mut min_dist = f32::MAX;
        let mut nearest = 0usize;

        for (i, centroid) in self.centroids.chunks(256).enumerate() {
            let dist = descriptor
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f32>();

            if dist < min_dist {
                min_dist = dist;
                nearest = i;
            }
        }

        // Accumulate residual
        let residual_start = nearest * 256;
        for (i, &v) in descriptor.iter().enumerate() {
            self.residuals[residual_start + i] += v - self.centroids[residual_start + i];
        }

        self.count += 1;
    }

    /// Finalize and return the aggregated vector (not normalized)
    pub fn finalize(self) -> Vec<f32> {
        self.residuals
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// Aggregate SuperPoint descriptors into a single vector using VLAD
///
/// # Arguments
/// * `descriptors` - flattened descriptor matrix [N, 256]
/// * `num_descriptors` - actual number of descriptors
/// * `max_descriptors` - maximum descriptors to use
/// * `centroids` - k-means centroids [K, 256]
///
/// # Returns
/// Aggregated vector [256], L2 normalized
pub fn vlad_aggregate(
    descriptors: &[f32],
    num_descriptors: usize,
    max_descriptors: usize,
    centroids: &[f32],
) -> Vec<f32> {
    let k = centroids.len() / 256;
    let mut aggregation = VladAggregation::with_centroids(centroids.to_vec(), k);

    // Use up to max_descriptors, selecting by index (evenly distributed)
    let step = if num_descriptors > max_descriptors {
        num_descriptors / max_descriptors
    } else {
        1
    };

    for i in (0..num_descriptors).step_by(step).take(max_descriptors) {
        let start = i * 256;
        let desc = &descriptors[start..start + 256];
        aggregation.add_descriptor(desc);
    }

    let mut vector = aggregation.finalize();

    // L2 normalize
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in &mut vector {
            *v /= norm;
        }
    }

    vector
}

/// Simple aggregation: just average all descriptors
/// Less accurate than VLAD but faster and works without k-means centroids
pub fn mean_aggregate(descriptors: &[f32], num_descriptors: usize) -> Vec<f32> {
    let dim = 256;
    let mut mean = vec![0.0f32; dim];

    for desc in descriptors.chunks(dim).take(num_descriptors) {
        for (i, &v) in desc.iter().enumerate() {
            mean[i] += v;
        }
    }

    let n = num_descriptors as f32;
    for m in &mut mean {
        *m /= n;
    }

    // L2 normalize
    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for m in &mut mean {
            *m /= norm;
        }
    }

    mean
}

/// Select top-k keypoints by score, keeping their corresponding descriptors
pub fn select_top_keypoints(
    keypoints: &[f32],      // [N, 2]
    scores: &[f32],         // [N]
    descriptors: &[f32],    // [N, 256]
    max_keypoints: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = keypoints.len() / 2;

    if n <= max_keypoints {
        return (keypoints.to_vec(), scores.to_vec(), descriptors.to_vec());
    }

    // Create index and sort by score
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    // Take top k
    let top_indices = &indices[..max_keypoints];

    let mut selected_kpts = Vec::with_capacity(max_keypoints * 2);
    let mut selected_scores = Vec::with_capacity(max_keypoints);
    let mut selected_desc = Vec::with_capacity(max_keypoints * 256);

    for &idx in top_indices {
        // Keypoint [x, y]
        selected_kpts.push(keypoints[idx * 2]);
        selected_kpts.push(keypoints[idx * 2 + 1]);
        // Score
        selected_scores.push(scores[idx]);
        // Descriptor
        let start = idx * 256;
        selected_desc.extend_from_slice(&descriptors[start..start + 256]);
    }

    (selected_kpts, selected_scores, selected_desc)
}

/// Normalize keypoints to [-1, 1] range for LightGlue
pub fn normalize_keypoints(keypoints: &[f32], width: u32, height: u32) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(keypoints.len());

    for (i, &k) in keypoints.iter().enumerate() {
        if i % 2 == 0 {
            // x coordinate
            normalized.push(2.0 * k / width as f32 - 1.0);
        } else {
            // y coordinate
            normalized.push(2.0 * k / height as f32 - 1.0);
        }
    }

    normalized
}

/// Initialize VLAD centroids using k-means++ algorithm
///
/// # Arguments
/// * `descriptors` - sample descriptors to initialize from [N, 256]
/// * `k` - number of centroids
/// * `max_iterations` - max k-means iterations
///
/// # Returns
/// Centroids [K, 256]
pub fn initialize_centroids(
    descriptors: &[f32],
    num_descriptors: usize,
    k: usize,
    max_iterations: usize,
) -> Vec<f32> {
    if num_descriptors == 0 || k == 0 {
        return vec![0.0; k * 256];
    }

    const DIM: usize = 256;
    let mut centroids = vec![0.0f32; k * DIM];

    // K-means++ initialization
    // 1. Choose first centroid randomly
    let first_idx = rand_idx(num_descriptors);
    for d in 0..DIM {
        centroids[d] = descriptors[first_idx * DIM + d];
    }

    // 2. Choose remaining centroids with probability proportional to distance
    for c in 1..k {
        // Compute distances to nearest centroid for each point
        let mut min_dists = vec![f32::MAX; num_descriptors];

        for i in 0..num_descriptors {
            let mut min_dist = f32::MAX;
            for j in 0..c {
                let dist = l2_dist(
                    &descriptors[i * DIM..i * DIM + DIM],
                    &centroids[j * DIM..j * DIM + DIM],
                );
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            min_dists[i] = min_dist;
        }

        // Choose next centroid with probability proportional to squared distance
        let total: f32 = min_dists.iter().map(|d| d * d).sum();
        let mut r = rand_float() * total;
        let mut chosen = 0;
        for (i, &d) in min_dists.iter().enumerate() {
            r -= d * d;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }

        // Copy chosen point as new centroid
        for d in 0..DIM {
            centroids[c * DIM + d] = descriptors[chosen * DIM + d];
        }
    }

    // Run k-means iterations
    let mut assignments = vec![0usize; num_descriptors];

    for _iter in 0..max_iterations {
        // Assign each descriptor to nearest centroid
        let mut changed = false;
        for i in 0..num_descriptors {
            let mut min_dist = f32::MAX;
            let mut nearest = 0;
            for c in 0..k {
                let dist = l2_dist(
                    &descriptors[i * DIM..i * DIM + DIM],
                    &centroids[c * DIM..c * DIM + DIM],
                );
                if dist < min_dist {
                    min_dist = dist;
                    nearest = c;
                }
            }
            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        for c in 0..k {
            let mut sum = vec![0.0f32; DIM];
            let mut count = 0usize;

            for i in 0..num_descriptors {
                if assignments[i] == c {
                    for d in 0..DIM {
                        sum[d] += descriptors[i * DIM + d];
                    }
                    count += 1;
                }
            }

            if count > 0 {
                for d in 0..DIM {
                    centroids[c * DIM + d] = sum[d] / count as f32;
                }
            }
        }
    }

    centroids
}

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let d = x - y;
        d * d
    }).sum::<f32>().sqrt()
}

fn rand_idx(n: usize) -> usize {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as usize) % n
}

fn rand_float() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_aggregate() {
        let descriptors = vec![1.0f32; 256 * 10];
        let result = mean_aggregate(&descriptors, 10);
        assert_eq!(result.len(), 256);
        // All values should be 1.0
        assert!((result[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_keypoints() {
        let kpts = vec![0.0, 0.0, 256.0, 256.0]; // 2 keypoints
        let normalized = normalize_keypoints(&kpts, 512, 512);
        assert_eq!(normalized, vec![-1.0, -1.0, 1.0, 1.0]);
    }
}
