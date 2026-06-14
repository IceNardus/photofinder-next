use std::sync::Arc;
use std::path::PathBuf;
use std::panic;
use anyhow::Result;
use tracing::{info, warn, error};
use tokio::sync::RwLock;
use image::{GenericImageView, RgbImage};

use crate::core::database::{Database, images::ImagesTable, faces::FacesTable, tasks::TasksTable, tasks::TaskStatus};
use crate::core::storage::FaceVectorStore;
use crate::core::clustering::PersonCluster;
use crate::ai::face::{FacePipeline, FaceDetector, FaceAligner, ArcFace, FaceFeature};
use crate::ai::image_classifier::ImageTypeClassifier;
use crate::search::object_search::{ObjectSearch, ObjectSearchConfig};

const BATCH_SIZE: usize = 100;
const MIN_QUALITY_THRESHOLD: f32 = 0.5;

/// CandidateRegion - a sliding window region, NOT an object
/// This is just an image patch for embedding, not a detected object
#[derive(Debug, Clone)]
pub struct CandidateRegion {
    pub bbox: [f32; 4], // [x, y, w, h] in pixels
    pub region_type: &'static str,
    pub size: u32,
}

pub struct ProcessingService {
    db: Arc<Database>,
    data_dir: PathBuf,
    face_pipeline: RwLock<Option<FacePipeline>>,
    face_store: Arc<FaceVectorStore>,
    person_cluster: RwLock<Option<PersonCluster>>,
    image_classifier: ImageTypeClassifier,
    // Object search for ROI-based object indexing
    object_search: RwLock<Option<ObjectSearch>>,
    // Shared processing stats for frontend display
    processing_stats: Arc<RwLock<crate::tauri_cmd::commands::ProcessingStats>>,
}

impl ProcessingService {
    pub fn new(db: Arc<Database>, data_dir: PathBuf, processing_stats: Arc<RwLock<crate::tauri_cmd::commands::ProcessingStats>>) -> Self {
        let face_store = Arc::new(FaceVectorStore::new(&data_dir));
        Self {
            db,
            data_dir,
            face_pipeline: RwLock::new(None),
            face_store,
            person_cluster: RwLock::new(None),
            image_classifier: ImageTypeClassifier::new(),
            object_search: RwLock::new(None),
            processing_stats,
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing ProcessingService AI models...");

        let models_dir = match crate::core::models::find_models_dir() {
            Some(dir) => dir,
            None => {
                return Err(anyhow::anyhow!(
                    "Models directory not found. Tried: exe path, data dir, and CWD"
                ));
            }
        };

        let scrfd_path = models_dir.join("scrfd_500m_bnkps.onnx");
        let arcface_path = models_dir.join("w600k_r50.onnx");

        info!("Loading SCRFD from: {}", scrfd_path.to_string_lossy());
        let detector = FaceDetector::new(&scrfd_path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("Failed to create detector: {}", e))?;
        info!("SCRFD loaded successfully");

        let aligner = FaceAligner::new();
        info!("FaceAligner created");

        info!("Loading ArcFace from: {}", arcface_path.to_string_lossy());
        let arcface = ArcFace::new(&arcface_path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("Failed to create arcface: {}", e))?;
        info!("ArcFace loaded successfully");

        let debug_dir = self.data_dir.join("debug");
std::fs::create_dir_all(&debug_dir).ok();
let pipeline = FacePipeline::with_debug(detector, aligner, arcface, &debug_dir);
        *self.face_pipeline.write().await = Some(pipeline);
        info!("FacePipeline initialized");

        let mut cluster = PersonCluster::new(Arc::clone(&self.db), &self.data_dir)?;
        let index_path = self.data_dir.join("vectors").join("person_cluster.bin");
        cluster.load_index(&index_path.to_string_lossy())?;
        *self.person_cluster.write().await = Some(cluster);
        info!("PersonCluster initialized");

        // Initialize face vector store
        self.face_store.init()
            .map_err(|e| anyhow::anyhow!("face_store.init failed: {}", e))?;
        info!("FaceVectorStore initialized with {} entries", self.face_store.count());

        // Initialize ObjectSearch models
        let mobileclip_path = models_dir.join("mobileclip_s2.onnx");
        let superpoint_path = models_dir.join("superpoint.onnx");
        let lightglue_path = models_dir.join("lightglue.onnx");

        if mobileclip_path.exists() && superpoint_path.exists() && lightglue_path.exists() {
            let config = ObjectSearchConfig::default();
            match ObjectSearch::new(
                &mobileclip_path.to_string_lossy(),
                &superpoint_path.to_string_lossy(),
                &lightglue_path.to_string_lossy(),
                config,
            ) {
                Ok(searcher) => {
                    // Initialize the HNSW index
                    let index_dir = self.data_dir.join("vectors");
                    if let Err(e) = searcher.init_index(&index_dir) {
                        warn!("[ObjectSearch] Failed to init index: {}", e);
                    }
                    *self.object_search.write().await = Some(searcher);
                    info!("ObjectSearch initialized");
                }
                Err(e) => {
                    warn!("[ObjectSearch] Failed to create: {}", e);
                }
            }
        } else {
            warn!("[ObjectSearch] Models not found, skipping initialization");
        }

        Ok(())
    }

    fn find_model_path(&self, model_name: &str) -> std::path::PathBuf {
        crate::core::models::find_model_path(model_name)
    }

    pub async fn start(&self) {
        info!("ProcessingService started");
        self.process_loop().await;
    }

    async fn process_loop(&self) {
        loop {
            match self.process_batch().await {
                Ok(0) => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
                Ok(count) => {
                    info!("Processed {} tasks", count);
                }
                Err(e) => {
                    error!("Processing error: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }
    }

    async fn process_batch(&self) -> Result<usize> {
        let tasks = TasksTable::get_pending(&self.db.conn, BATCH_SIZE)?;
        if tasks.is_empty() {
            return Ok(0);
        }

        info!("Processing batch of {} tasks", tasks.len());
        let mut processed = 0;

        for task in &tasks {
            match self.process_task(task.image_id).await {
                Ok(_) => {
                    TasksTable::update_status(&self.db.conn, task.id, TaskStatus::Completed)?;
                    ImagesTable::update_scan_status(&self.db.conn, task.image_id, "completed")?;
                    processed += 1;
                }
                Err(e) => {
                    warn!("Failed to process image_id {}: {}", task.image_id, e);
                    TasksTable::update_status(&self.db.conn, task.id, TaskStatus::Failed)?;
                }
            }
        }

        Ok(processed)
    }

       async fn process_task(&self, image_id: i64) -> Result<()> {
        let image = ImagesTable::get_by_id(&self.db.conn, image_id)?
            .ok_or_else(|| anyhow::anyhow!("Image not found: {}", image_id))?;

        // Update processing stats for frontend display
        {
            let mut stats = self.processing_stats.write().await;
            stats.current_image = image.path.clone();
            stats.current_faces = 0;
            stats.current_patches = 0;
            stats.current_objects = 0;
            stats.log_message = format!("扫描: {}", std::path::Path::new(&image.path).file_name().and_then(|s| s.to_str()).unwrap_or(&image.path));
        }

        // Skip non-photo images using ImageTypeClassifier
        let (img_type, reason) = self.image_classifier.classify_path(std::path::Path::new(&image.path));
        if !img_type.should_process() {
            ImagesTable::update_scan_status(&self.db.conn, image_id, "completed")?;
            return Ok(());
        }

        // Process faces only (patches and objects disabled)
        self.process_faces(image_id).await;

        // Process objects (ROI extraction + MobileCLIP embedding + HNSW indexing)
        self.process_objects(image_id).await;

        // Final log message with all counts
        let final_stats = self.processing_stats.read().await;
        let filename = std::path::Path::new(&image.path).file_name().and_then(|s| s.to_str()).unwrap_or(&image.path).to_string();
        let completion_msg = format!("完成: {} | 人脸{} 物品{}", filename, final_stats.current_faces, final_stats.current_objects);
        drop(final_stats);  // Release read lock before write
        {
            let mut stats = self.processing_stats.write().await;
            stats.log_message = completion_msg.clone();
            stats.last_completion_message = completion_msg;
        }

        Ok(())
    }

    async fn process_faces(&self, image_id: i64) {
        let pipeline_guard = self.face_pipeline.read().await;
        let pipeline = match pipeline_guard.as_ref() {
            Some(p) => p,
            None => {
                warn!("FacePipeline not initialized, skipping faces for image {}", image_id);
                return;
            }
        };

        let image = match ImagesTable::get_by_id(&self.db.conn, image_id) {
            Ok(Some(img)) => img,
            Ok(None) => {
                warn!("Image {} not found", image_id);
                return;
            }
            Err(e) => {
                warn!("Failed to get image {}: {}", image_id, e);
                return;
            }
        };

        eprintln!("[FACE] Processing faces for image_id: {}", image_id);

        let features = match pipeline.process_image(&image.path) {
            Ok(f) => {
                eprintln!("[FACE] SCRFD detection returned {} features", f.len());
                f
            }
            Err(e) => {
                eprintln!("[FACE] Face detection FAILED for {}: {}", image_id, e);
                warn!("[FACE] Face detection failed for image {}: {}", image_id, e);
                return;
            }
        };

        info!("[FACE] Found {} faces in image {}", features.len(), image_id);

        // Update processing stats
        {
            let mut stats = self.processing_stats.write().await;
            stats.current_faces = features.len();
        }

        for feature in features {
            if let Err(e) = self.process_face(image_id, &feature).await {
                warn!("[FACE] Failed to process face in image {}: {}", image_id, e);
            }
        }
    }

    async fn process_objects(&self, image_id: i64) {
        let searcher_guard = self.object_search.read().await;
        let searcher = match searcher_guard.as_ref() {
            Some(s) => s,
            None => {
                // ObjectSearch not initialized, skip
                return;
            }
        };

        let image = match ImagesTable::get_by_id(&self.db.conn, image_id) {
            Ok(Some(img)) => img,
            Ok(None) => {
                warn!("Image {} not found", image_id);
                return;
            }
            Err(e) => {
                warn!("Failed to get image {}: {}", image_id, e);
                return;
            }
        };

        // Skip if file doesn't exist
        if !std::path::Path::new(&image.path).exists() {
            warn!("Image file not found: {}", image.path);
            return;
        }

        // Load image for ROI extraction and embedding
        let img = match image::open(&image.path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                warn!("[OBJ] Failed to open {}: {}", image.path, e);
                return;
            }
        };

        // Use multi-scale sliding window ROI instead of selective search
        // min_size=32, stride_ratio=0.5, max_rois computed dynamically
        let (img_w, img_h) = img.dimensions();
        eprintln!("[OBJ] process_objects: image_id={}, path={}, img_dims={}x{}", image_id, image.path, img_w, img_h);

        let rois = self.multi_scale_roi(&img, 32, 0.5, 0);

        if rois.is_empty() {
            info!("[OBJ] No ROIs generated for image {}", image_id);
            return;
        }

        eprintln!("[OBJ] Generated {} ROIs for image {} (first ROI: {:?})", rois.len(), image_id, rois.first().map(|r| r.bbox));
        info!("[OBJ] Generated {} ROIs for image {}", rois.len(), image_id);

        let mut object_count = 0;

        for roi in &rois {
            // Crop ROI from image
            let [x, y, w, h] = roi.bbox;
            let x1 = x as u32;
            let y1 = y as u32;
            let x2 = (x1 + w as u32).min(img.width());
            let y2 = (y1 + h as u32).min(img.height());

            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            eprintln!("[OBJ] Cropping ROI: ({},{}) {}x{} from {}x{} image",
                x1, y1, x2-x1, y2-y1, img.width(), img.height());

            let crop = image::DynamicImage::ImageRgb8(img.clone()).crop_imm(x1, y1, x2 - x1, y2 - y1);
            match searcher.extract_embedding(&crop) {
                Ok(embedding) => {
                    if let Err(e) = searcher.add_to_index(image_id, &image.path, roi.region_type, roi.bbox, &embedding) {
                        warn!("[OBJ] Failed to add to index: {}", e);
                    } else {
                        object_count += 1;
                    }
                }
                Err(e) => {
                    warn!("[OBJ] Embedding failed: {}", e);
                }
            }
        }

        // Update processing stats
        {
            let mut stats = self.processing_stats.write().await;
            stats.current_objects = object_count;
        }

        info!("[OBJ] Indexed {} objects from image {}", object_count, image_id);
    }

    /// Multi-scale sliding window ROI extraction with adaptive ROI count
    fn multi_scale_roi(&self, img: &RgbImage, min_size: u32, stride_ratio: f32, _max_rois: usize) -> Vec<CandidateRegion> {
        let (width, height) = img.dimensions();
        let img_area = (width * height) as f32;
        let mut regions = Vec::new();

        // Dynamic ROI count based on image area
        // Small image: 10-20, Medium: 30-50, Large: 60-100
        let dynamic_max_rois = ((img_area / 50_000.0) as usize).clamp(10, 100);

        let scales = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0];

        for &scale in &scales {
            let win_w = (width as f32 * scale) as u32;
            let win_h = (height as f32 * scale) as u32;

            if win_w < min_size || win_h < min_size {
                continue;
            }

            let stride_x = std::cmp::max(1, (win_w as f32 * stride_ratio) as u32);
            let stride_y = std::cmp::max(1, (win_h as f32 * stride_ratio) as u32);

            let mut y = 0;
            while y + win_h <= height {
                let mut x = 0;
                while x + win_w <= width {
                    regions.push(CandidateRegion {
                        bbox: [x as f32, y as f32, win_w as f32, win_h as f32],
                        region_type: "sliding_window",
                        size: win_w * win_h,
                    });
                    x += stride_x;
                }
                y += stride_y;
            }
        }

        // Add full image ROI
        regions.push(CandidateRegion {
            bbox: [0.0, 0.0, width as f32, height as f32],
            region_type: "full_image",
            size: width * height,
        });

        // Sort by size and truncate to dynamic count
        regions.sort_by(|a, b| b.size.cmp(&a.size));
        regions.truncate(dynamic_max_rois);

        eprintln!("[ROI] multi_scale_roi: {}x{}, dynamic_max_rois={}, final={}",
                  width, height, dynamic_max_rois, regions.len());

        regions
    }

    async fn process_face(&self, image_id: i64, feature: &FaceFeature) -> Result<()> {
        let embedding = &feature.embedding;

        // Verify image exists first
        if ImagesTable::get_by_id(&self.db.conn, image_id)?.is_none() {
            return Err(anyhow::anyhow!("Image {} not found, cannot process face", image_id));
        }

        eprintln!("[FACE_STORE] process_face: image_id={}, embedding_len={}", image_id, embedding.len());
        eprintln!("[FACE_STORE] process_face: face_store index_path={}", self.face_store.index_path().display());
        eprintln!("[FACE_STORE] process_face: face_store index_path exists={}", self.face_store.index_path().exists());
        eprintln!("[FACE_STORE] process_face: data_dir={}", self.data_dir.display());

        // Save vector first (this might fail, but at least we don't orphan data)
        let loc = match self.face_store.save(0, embedding) {
            Ok(l) => {
                eprintln!("[FACE_STORE] process_face: save succeeded, offset={}, length={}", l.offset, l.length);
                l
            },
            Err(e) => {
                eprintln!("[FACE_STORE] process_face: save FAILED: {}", e);
                return Err(anyhow::anyhow!("face_store.save failed: {}", e));
            }
        };
        let vector_offset = loc.offset as i64;
        let vector_length = loc.vector_length as i32;

        // Now add face to database with the vector info
        let face_id = match FacesTable::add(
            &self.db.conn,
            image_id,
            &feature.bbox,
            feature.detector_score,
            feature.blur_score,
            feature.pose_score,
            feature.face_area_score,
            feature.quality,
            0.0,
            0.0,
            0.0,
            vector_offset,
            vector_length,
        ) {
            Ok(id) => id,
            Err(e) => {
                // Cleanup: delete the vector we just saved
                eprintln!("[FACE] Failed to add face to DB, deleting orphaned vector at offset {}: {}", loc.offset, e);
                let _ = self.face_store.delete_vector(loc.offset);
                return Err(anyhow::anyhow!("Failed to save face: {}", e));
            }
        };

        // Update the vector entry with the correct face_id
        self.face_store.update_id(loc.offset, face_id)?;

        info!("Saved face {} for image {}", face_id, image_id);

        self.assign_person(face_id, embedding).await?;

        Ok(())
    }

    async fn assign_person(&self, face_id: i64, embedding: &[f32]) -> Result<()> {
        let mut cluster_guard = self.person_cluster.write().await;
        let cluster = cluster_guard.as_mut()
            .ok_or_else(|| anyhow::anyhow!("PersonCluster not initialized"))?;

        match cluster.assign_person(embedding) {
            Ok(person_id) => {
                info!("Face {} assigned to existing person {}", face_id, person_id);
                cluster.add_face_to_person(face_id, person_id, embedding)?;
            }
            Err(_) => {
                info!("Creating new person for face {}", face_id);
                cluster.create_person(face_id, embedding)?;
            }
        }

        Ok(())
    }
}

pub fn start_processing_service(db: Arc<Database>, data_dir: PathBuf, processing_stats: Arc<RwLock<crate::tauri_cmd::commands::ProcessingStats>>) {
    eprintln!("[PROCESSING] start_processing_service ENTRY");
    let service = Arc::new(ProcessingService::new(db, data_dir, processing_stats));
    eprintln!("[PROCESSING] service created");
    let service_clone = service.clone();
    eprintln!("[PROCESSING] service_clone created");

    let handle = std::thread::spawn(move || {
        eprintln!("[PROCESSING] >>> thread spawned, creating tokio runtime");
        let rt = tokio::runtime::Runtime::new().unwrap();
        eprintln!("[PROCESSING] >>> tokio runtime created, entering block_on");
        rt.block_on(async {
            eprintln!("[PROCESSING] >>> inside block_on, calling initialize...");
            info!(">>> ProcessingService thread starting, calling initialize...");
            match service_clone.initialize().await {
                Ok(_) => {
                    eprintln!("[PROCESSING] >>> initialize succeeded!");
                    info!(">>> ProcessingService initialized successfully!");
                    service_clone.start().await;
                }
                Err(e) => {
                    eprintln!("[PROCESSING] >>> initialize FAILED: {}", e);
                    error!(">>> FAILED to initialize ProcessingService: {}", e);
                }
            }
        });
        eprintln!("[PROCESSING] >>> block_on ended");
    });
    eprintln!("[PROCESSING] >>> thread handle created, id: {:?}", handle.thread().id());
    info!(">>> start_processing_service spawns thread");
}