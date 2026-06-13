//! ROI (Region of Interest) Extractor using Selective Search
//!
//! Based on paper: "Selective Search for Object Recognition" (Uijlings et al., 2012)
//! Modified for efficiency in Rust

use image::{GenericImageView, RgbImage, GrayImage};
use std::collections::HashMap;

/// A region found by selective search
#[derive(Debug, Clone)]
pub struct Region {
    pub bbox: [f32; 4],      // [x, y, w, h] in pixels
    pub region_type: &'static str,
    pub size: u32,            // region size in pixels
}

/// Perform selective search on an RGB image
pub fn selective_search(img: &RgbImage, max_regions: usize) -> Vec<Region> {
    let (width, height) = img.dimensions();
    let mut regions = Vec::new();

    // Step 1: ComputeFelzenszwalbHuttenlocher segmentation
    let segments = felzenszwalb_segmentation(img);

    // Step 2: Get initial regions from segmentation
    let mut region_map: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for y in 0..height {
        for x in 0..width {
            let seg_id = segments[(y * width + x) as usize];
            region_map.entry(seg_id).or_insert_with(Vec::new).push((x, y));
        }
    }

    // Step 3: Create regions from segments
    for (seg_id, pixels) in region_map {
        if pixels.is_empty() {
            continue;
        }

        let min_x = pixels.iter().map(|p| p.0).min().unwrap();
        let max_x = pixels.iter().map(|p| p.0).max().unwrap();
        let min_y = pixels.iter().map(|p| p.1).min().unwrap();
        let max_y = pixels.iter().map(|p| p.1).max().unwrap();

        let bbox = [
            min_x as f32,
            min_y as f32,
            (max_x - min_x + 1) as f32,
            (max_y - min_y + 1) as f32,
        ];

        regions.push(Region {
            bbox,
            region_type: "segment",
            size: pixels.len() as u32,
        });
    }

    // Step 4: Hierarchical grouping (simplified)
    let merged = hierarchical_grouping(&mut regions, &segments, width, height);

    // Step 5: Filter and sort by size
    let merged_count = merged.len();
    let mut final_regions: Vec<Region> = merged
        .into_iter()
        .filter(|r| r.size >= 500)  // Minimum region size
        .collect();

    let filtered_count = final_regions.len();
    final_regions.sort_by(|a, b| b.size.cmp(&a.size));
    let before_truncate = final_regions.len();
    final_regions.truncate(max_regions);

    // Add image boundary region
    final_regions.push(Region {
        bbox: [0.0, 0.0, width as f32, height as f32],
        region_type: "full_image",
        size: (width * height) as u32,
    });

    // Log stats
    eprintln!("[ROI] selective_search: img={}x{}, merged={}, size_filtered={}, truncated={}/{}+1, final={}",
             width, height, merged_count, filtered_count, before_truncate, max_regions, final_regions.len());

    final_regions
}

/// Felzenszwalb-Huttenlocher segmentation
fn felzenszwalb_segmentation(img: &RgbImage) -> Vec<u32> {
    let (width, height) = img.dimensions();
    let num_pixels = (width * height) as usize;

    // Initialize disjoint set
    let mut parent = (0..num_pixels as u32).collect::<Vec<_>>();
    let mut rank = vec![0u32; num_pixels];

    // Build gradient image
    let gradient = compute_gradient(img);

    // Initialize each pixel as its own region
    let mut thresholds = vec![30.0f32; num_pixels];  // Initial threshold k=30

    // Sort edges by gradient difference
    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;

            // Horizontal edge
            let idx2 = (y * width + x + 1) as usize;
            let diff_h = (gradient[idx] - gradient[idx2]).abs();
            edges.push((idx as u32, idx2 as u32, diff_h));

            // Vertical edge
            let idx3 = ((y + 1) * width + x) as usize;
            let diff_v = (gradient[idx] - gradient[idx3]).abs();
            edges.push((idx as u32, idx3 as u32, diff_v));
        }
    }

    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    fn find(parent: &[u32], x: u32) -> u32 {
        if parent[x as usize] != x {
            find(parent, parent[x as usize])
        } else {
            x
        }
    }

    fn union(parent: &mut [u32], rank: &mut [u32], x: u32, y: u32) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px == py { return; }

        if rank[px as usize] < rank[py as usize] {
            parent[px as usize] = py;
        } else if rank[px as usize] > rank[py as usize] {
            parent[py as usize] = px;
        } else {
            parent[py as usize] = px;
            rank[px as usize] += 1;
        }
    }

    // Process edges
    for (a, b, diff) in edges {
        let pa = find(&parent, a);
        let pb = find(&parent, b);
        if pa != pb {
            let threshold = thresholds[pa as usize].min(thresholds[pb as usize]) + diff;
            if diff <= threshold {
                union(&mut parent, &mut rank, pa, pb);
                let new_parent = find(&parent, pa);
                thresholds[new_parent as usize] = threshold;
            }
        }
    }

    // Renumber segments
    let mut segment_map = HashMap::new();
    let mut next_id = 0u32;
    let mut result = vec![0u32; num_pixels];

    for i in 0..num_pixels {
        let p = find(&parent, i as u32);
        let seg_id = *segment_map.entry(p).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        result[i] = seg_id;
    }

    result
}

/// Compute gradient magnitude for segmentation
fn compute_gradient(img: &RgbImage) -> Vec<f32> {
    let (width, height) = img.dimensions();
    let mut gradient = vec![0.0f32; (width * height) as usize];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;

            // Sobel-like gradient
            let p00 = img.get_pixel(x - 1, y - 1);
            let p10 = img.get_pixel(x, y - 1);
            let p20 = img.get_pixel(x + 1, y - 1);
            let p01 = img.get_pixel(x - 1, y);
            let p21 = img.get_pixel(x + 1, y);
            let p02 = img.get_pixel(x - 1, y + 1);
            let p12 = img.get_pixel(x, y + 1);
            let p22 = img.get_pixel(x + 1, y + 1);

            // Convert to luminance
            let lum = |p: &image::Rgb<u8>| -> f32 {
                0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
            };

            // Gradient in x and y directions
            let gx = -lum(p00) - 2.0 * lum(p01) - lum(p02) + lum(p20) + 2.0 * lum(p21) + lum(p22);
            let gy = -lum(p00) - 2.0 * lum(p10) - lum(p20) + lum(p02) + 2.0 * lum(p12) + lum(p22);

            gradient[idx] = (gx * gx + gy * gy).sqrt();
        }
    }

    gradient
}

/// Simplified hierarchical grouping
fn hierarchical_grouping(
    regions: &mut Vec<Region>,
    _segments: &[u32],
    width: u32,
    height: u32,
) -> Vec<Region> {
    // For simplicity, we'll keep the largest regions
    // A full implementation would merge neighboring regions by color/texture similarity

    let total_pixels = width * height;

    // Keep regions that are between 5% and 80% of image size
    regions.retain(|r| {
        let size_ratio = r.size as f32 / total_pixels as f32;
        size_ratio >= 0.001 && size_ratio <= 0.95
    });

    // Add some multi-scale boxes
    let scales = [0.2, 0.4, 0.6, 0.8];
    for &scale in &scales {
        let w = (width as f32 * scale) as u32;
        let h = (height as f32 * scale) as u32;
        let x = (width - w) / 2;
        let y = (height - h) / 2;

        regions.push(Region {
            bbox: [x as f32, y as f32, w as f32, h as f32],
            region_type: "multi_scale",
            size: w * h,
        });
    }

    regions.sort_by(|a, b| b.size.cmp(&a.size));
    regions.dedup_by(|a, b| {
        // Remove very similar regions
        (a.bbox[0] - b.bbox[0]).abs() < 20.0
            && (a.bbox[1] - b.bbox[1]).abs() < 20.0
            && (a.bbox[2] - b.bbox[2]).abs() < 20.0
            && (a.bbox[3] - b.bbox[3]).abs() < 20.0
    });

    regions.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_search() {
        let img = RgbImage::new(100, 100);
        let regions = selective_search(&img, 10);
        assert!(!regions.is_empty());
    }
}