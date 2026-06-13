use rusqlite::{Connection, params};
use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use tracing::info;

pub mod images;
pub mod faces;
pub mod tasks;
pub mod persons;
pub mod patches;

use images::ImagesTable;
use faces::FacesTable;
use tasks::TasksTable;
use persons::PersonsTable;
use patches::{PatchFeaturesTable, PatchVectorsTable};

pub struct Database {
    pub conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "[DB] init_schema ENTRY");
        std::io::stderr().flush().ok();

        let conn = self.conn.lock().unwrap();
        let _ = writeln!(std::io::stderr(), "[DB] conn.lock() acquired");
        std::io::stderr().flush().ok();
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified_time INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                thumbnail_path TEXT,
                thumbnail_status TEXT NOT NULL DEFAULT 'pending',
                scan_status TEXT NOT NULL DEFAULT 'pending',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT NOT NULL UNIQUE,
                file_count INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scan_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                task_type TEXT NOT NULL DEFAULT 'image',
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id)
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                person_id INTEGER,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                detector_score REAL NOT NULL,
                blur_score REAL NOT NULL,
                pose_score REAL NOT NULL,
                face_area_score REAL NOT NULL,
                quality REAL NOT NULL,
                yaw REAL,
                pitch REAL,
                roll REAL,
                vector_offset INTEGER NOT NULL,
                vector_length INTEGER NOT NULL,
                storage_status TEXT NOT NULL DEFAULT 'pending',
                index_status TEXT NOT NULL DEFAULT 'pending',
                FOREIGN KEY (image_id) REFERENCES images(id)
            );

            CREATE INDEX IF NOT EXISTS idx_images_hash ON images(hash);
            CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
            CREATE INDEX IF NOT EXISTS idx_images_scan_status ON images(scan_status);
            CREATE INDEX IF NOT EXISTS idx_duplicate_groups_hash ON duplicate_groups(hash);
            CREATE INDEX IF NOT EXISTS idx_scan_tasks_status ON scan_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_scan_tasks_type ON scan_tasks(task_type);
            CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
            CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);
            CREATE INDEX IF NOT EXISTS idx_faces_storage_status ON faces(storage_status);
            CREATE INDEX IF NOT EXISTS idx_faces_index_status ON faces(index_status);
        "#)?;
        let _ = writeln!(std::io::stderr(), "[DB] execute_batch completed");
        std::io::stderr().flush().ok();
        // Create persons table directly (don't call PersonsTable::create which would deadlock)
        let _ = writeln!(std::io::stderr(), "[DB] creating persons table...");
        std::io::stderr().flush().ok();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_count INTEGER NOT NULL DEFAULT 0,
                cover_face_id INTEGER,
                center_vector_offset INTEGER NOT NULL,
                center_vector_length INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;
        let _ = writeln!(std::io::stderr(), "[DB] persons table created");
        std::io::stderr().flush().ok();
        conn.execute("CREATE INDEX IF NOT EXISTS idx_persons_face_count ON persons(face_count)", [])?;
        let _ = writeln!(std::io::stderr(), "[DB] persons index created");
        std::io::stderr().flush().ok();

        // Create patch_features table for SuperPoint features
        let _ = writeln!(std::io::stderr(), "[DB] creating patch_features table...");
        std::io::stderr().flush().ok();
        PatchFeaturesTable::create(&conn)?;
        let _ = writeln!(std::io::stderr(), "[DB] patch_features table created");
        std::io::stderr().flush().ok();

        // Migrate existing data to add color_hist column if needed
        let _ = writeln!(std::io::stderr(), "[DB] migrating patch_features...");
        std::io::stderr().flush().ok();
        PatchFeaturesTable::migrate(&conn)?;
        let _ = writeln!(std::io::stderr(), "[DB] patch_features migration done");
        std::io::stderr().flush().ok();

        // Create patch_vectors table for HNSW indexing
        let _ = writeln!(std::io::stderr(), "[DB] creating patch_vectors table...");
        std::io::stderr().flush().ok();
        PatchVectorsTable::create(&conn)?;
        let _ = writeln!(std::io::stderr(), "[DB] patch_vectors table created");
        std::io::stderr().flush().ok();

        drop(conn);
        let _ = writeln!(std::io::stderr(), "[DB] conn.lock() released");
        std::io::stderr().flush().ok();
        info!("Database schema initialized");
        let _ = writeln!(std::io::stderr(), "[DB] init_schema returning OK");
        std::io::stderr().flush().ok();
        Ok(())
    }

    pub fn clear_all(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        // Enable foreign keys to ensure proper deletion order
        conn.execute("PRAGMA foreign_keys = ON", [])?;
        conn.execute_batch(r#"
            DELETE FROM patch_vectors;
            DELETE FROM patch_features;
            DELETE FROM objects;
            DELETE FROM faces;
            DELETE FROM scan_tasks;
            DELETE FROM duplicate_groups;
            DELETE FROM images;
        "#)?;
        // Disable foreign keys after operation
        conn.execute("PRAGMA foreign_keys = OFF", [])?;
        info!("Database cleared");
        Ok(())
    }

    pub fn begin_transaction(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("BEGIN TRANSACTION", [])?;
        Ok(())
    }

    pub fn commit_transaction(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("COMMIT", [])?;
        Ok(())
    }

    pub fn rollback_transaction(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("ROLLBACK", [])?;
        Ok(())
    }
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self {
            conn: Arc::clone(&self.conn),
        }
    }
}

// Safety: Database wraps Connection in Arc<Mutex>, so it is safe to send across threads
// The Mutex ensures only one thread can access the connection at a time
unsafe impl Send for Database {}
unsafe impl Sync for Database {}