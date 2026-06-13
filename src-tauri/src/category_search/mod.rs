//! Category-based object search using YOLOv8 + MobileCLIP + HNSW
//!
//! Architecture:
//! 1. Indexing: YOLOv8 detects objects -> MobileCLIP embeds -> HNSW index
//! 2. Search: User provides reference images -> Generate prototype -> Search HNSW
//!
//! This is a simplified alternative to patch-based search (SuperPoint + LightGlue)

pub mod detector;
pub mod embedder;
pub mod index;
pub mod search;

pub use detector::YoloV8Detector;
pub use embedder::MobileClipEmbedder;
pub use index::CategoryIndex;
pub use search::{CategorySearch, CategorySearchConfig, SearchResult};