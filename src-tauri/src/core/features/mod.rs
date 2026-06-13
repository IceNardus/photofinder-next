//! Feature extraction and matching module
//!
//! Contains:
//! - Patch data structures
//! - Descriptor aggregation (VLAD)
//! - SuperPoint feature extraction
//! - LightGlue feature matching

pub mod patch;
pub mod aggregation;

pub use patch::{
    Bbox, PatchFeature, PatchVector, DescriptorStats, PatchConfig, Patch, ImageFeatures,
    SplitPatch, split_into_patches, extract_patch, early_exit_check,
    compute_color_histogram, histogram_intersection,
    PATCH_SIZE, STRIDE,
};
pub use aggregation::{
    vlad_aggregate, mean_aggregate, select_top_keypoints, normalize_keypoints, VladAggregation,
    initialize_centroids,
};
