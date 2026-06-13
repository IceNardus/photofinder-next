//! Object Index Module
//!
//! Provides HNSW-based object search with the following architecture:
//!
//! [Scanning] Image → ROI Extraction → MobileCLIP embedding → object_vectors.bin → HNSW
//!
//! [Query] Query ROI → MobileCLIP embedding → HNSW Top-N objects
//!       → Aggregate by image_id → Top-N images
//!       → SuperPoint + LightGlue rerank → Final Top-K

pub mod vector_store;
pub mod hnsw_index;

pub use vector_store::{VectorStore, VectorMeta};
pub use hnsw_index::{ObjectHnswIndex, ObjectSearchResult};