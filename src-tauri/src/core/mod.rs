pub mod scanner;
pub mod queue;
pub mod pipeline;
pub mod index;
pub mod database;
pub mod storage;
pub mod statistics;
pub mod thumbnail;
pub mod processing;
pub mod clustering;
pub mod image_filter;
pub mod features;
pub mod models;
pub mod feature_extractor;
pub mod geometry;

pub use scanner::Scanner;
pub use queue::TaskQueue;
pub use pipeline::{PersonPipeline, WorkerPool, Dispatcher};
pub use index::FaceIndex;
pub use database::Database;
pub use storage::FaceVectorStore;
pub use statistics::StatisticsCollector;
pub use thumbnail::{ThumbnailStore, ThumbnailWorker};
pub use clustering::PersonCluster;
pub use processing::start_processing_service;
pub use image_filter::should_skip_for_face_pipeline;
pub use feature_extractor::{FeatureExtractor, save_image_features};
pub use features::{
    Bbox, PatchFeature, PatchVector, DescriptorStats, PatchConfig, Patch, ImageFeatures,
    vlad_aggregate, mean_aggregate, select_top_keypoints, normalize_keypoints,
};