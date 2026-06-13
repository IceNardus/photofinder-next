# PhotoFinder Next - Technical Specification

## 1. Project Overview

- **Name**: PhotoFinder Next
- **Type**: Desktop Application (Local Offline Photo Retrieval System)
- **Core Features**: Person Search + Object Search
- **Target Scale**: 1M images, 5M faces, 20M objects
- **Query Response**: < 100ms
- **Platforms**: macOS, Windows

## 2. Architecture Principles

### Principle 1: Business-Driven Architecture
- **Wrong**: MobileNet → ArcFace (model-centric)
- **Correct**: Person Search / Object Search (business-centric)
- Each business owns its independent Pipeline

### Principle 2: Independent Feature Storage
- Face Vector ≠ Object Vector (NO fusion)
- Independent indexes: face_index.bin, object_index.bin
- No cross-contamination of embeddings

### Principle 3: Scan-Inference Decoupling
- Scanner: discovers files only
- Inference: background tasks
- Database: state management

## 3. System Architecture

```
PhotoFinder Next
├── Scanner         (file discovery)
├── Task Queue      (orchestration)
├── Face Pipeline  (SCRFD → Quality → ArcFace)
├── Object Pipeline (YOLO → MobileNetV4)
├── Vector Index   (HNSW)
├── SQLite         (metadata)
└── Search API     (Tauri Commands)
```

## 4. Module Structure

```
src-tauri/
├── src/
│   ├── core/
│   │   ├── scanner/       # File discovery
│   │   ├── queue/         # Task orchestration
│   │   ├── pipeline/     # Processing pipelines
│   │   ├── index/        # Vector indexing (HNSW)
│   │   └── database/     # SQLite operations
│   ├── ai/
│   │   ├── face/
│   │   │   ├── detector.rs    # SCRFD
│   │   │   ├── align.rs       # Face alignment
│   │   │   ├── quality.rs     # Quality scoring
│   │   │   └── arcface.rs     # Embedding extraction
│   │   └── object/
│   │       ├── detector.rs    # YOLO
│   │       └── mobilenet.rs   # Embedding extraction
│   ├── search/
│   │   ├── person_search.rs
│   │   └── object_search.rs
│   └── tauri/
│       └── commands.rs
├── resources/
│   └── models/
│       ├── scrfd_500m_bnkps.onnx
│       ├── w600k_r50.onnx          # ArcFace
│       ├── yolov8n.onnx
│       └── mobilenetv4_conv_small_e2400_1024.onnx
├── data/                          # Runtime data
│   ├── vectors/
│   │   ├── face/
│   │   └── object/
│   ├── face_index.bin
│   └── object_index.bin
└── src-tauri.conf.json
```

## 5. Database Schema

### images
```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    hash TEXT UNIQUE NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

### scan_tasks
```sql
CREATE TABLE scan_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    task_type TEXT NOT NULL,  -- 'face' | 'object'
    status TEXT NOT NULL,    -- 'pending' | 'processing' | 'completed' | 'failed'
    retry_count INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id)
);
```

### faces
```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    detector_score REAL NOT NULL,
    quality REAL NOT NULL,
    yaw REAL,
    pitch REAL,
    roll REAL,
    embedding_path TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id)
);
```

### objects
```sql
CREATE TABLE objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    embedding_path TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id)
);
```

## 6. Data Flow

### Face Pipeline
```
Image
  → SCRFD (detector)
  → Face Quality Filter (≥ 0.45)
  → ArcFace (embedding)
  → face_index.bin (HNSW)
```

### Object Pipeline
```
Image
  → YOLOv8 (detector)
  → ROI Crop
  → MobileNetV4 (embedding)
  → object_index.bin (HNSW)
```

## 7. Face Quality Scoring

```
quality = 0.30 × detector_score
        + 0.25 × face_area_score
        + 0.25 × blur_score
        + 0.20 × pose_score

Threshold: quality ≥ 0.45
```

## 8. Vector Storage

- SQLite stores paths only, NOT vectors
- Directory structure:
  ```
  data/vectors/face/{id}.vec
  data/vectors/object/{id}.vec
  ```

## 9. Vector Index

- Algorithm: HNSW
- Metric: Cosine Similarity
- Independent indexes for face and object

## 10. Concurrency Architecture

```
Scanner Thread
    ↓
Worker Pool
    ├── Face Worker × N
    ├── Object Worker × N
    ├── Index Worker × 1
    └── Database Worker × 1
```

## 11. Tauri Commands

| Command | Description |
|---------|-------------|
| `scan_folder` | Start scanning a folder |
| `stop_scan` | Stop ongoing scan |
| `clear_database` | Clear all data |
| `rebuild_index` | Rebuild vector index |
| `search_person` | Search by face |
| `search_object` | Search by object |
| `get_scan_status` | Get scan progress |
| `get_statistics` | Get database statistics |

## 12. Frontend Pages

### Page 1: Scan
- Add/remove scan directories
- Start/stop scan
- Rebuild index

### Page 2: Person Search
- Upload reference image
- Run retrieval
- Results in waterfall layout

### Page 3: Object Search
- Upload reference image
- Select detected object
- Run retrieval
- Results in waterfall layout

## 13. Out of Scope (V1)

- Video analysis
- CLIP / OCR / Scene recognition
- Similar image search
- Online services / User system
- Tags / Cloud sync

## 14. V2 Extensions (Future)

- Person clustering
- Person naming
- Pet recognition
- Video keyframe analysis
- OCR
- CLIP semantic search