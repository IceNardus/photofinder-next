//! Database operations for patch features (SuperPoint keypoints + descriptors)

use rusqlite::{params, Connection, Result};
use std::sync::Mutex;
use std::path::Path;

use crate::core::features::{PatchFeature, PatchVector, Bbox};

/// Store complete SuperPoint features for patches
pub struct PatchFeaturesTable;

impl PatchFeaturesTable {
    pub const TABLE_NAME: &'static str = "patch_features";

    pub fn create(conn: &Connection) -> Result<()> {
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS patch_features (
                patch_id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                patch_index INTEGER NOT NULL,
                keypoints BLOB NOT NULL,
                descriptors BLOB NOT NULL,
                num_keypoints INTEGER NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_w REAL NOT NULL,
                bbox_h REAL NOT NULL,
                color_hist BLOB NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
            "#,
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_features_image_id ON patch_features(image_id)",
            [],
        )?;

        Ok(())
    }

    /// Migrate existing database to add color_hist column
    pub fn migrate(conn: &Connection) -> Result<()> {
        // Check if color_hist column exists
        let has_color_hist: bool = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('patch_features') WHERE name='color_hist'",
            [],
            |row| row.get::<_, i64>(0),
        )? > 0;

        if !has_color_hist {
            // Add column with NULL default first
            conn.execute(
                "ALTER TABLE patch_features ADD COLUMN color_hist BLOB",
                [],
            )?;

            // Update existing rows to set color_hist to zeros
            let zero_hist = vec![0.0f32; 64];
            let zero_bytes = f32_to_bytes(&zero_hist);
            conn.execute(
                "UPDATE patch_features SET color_hist = ?1 WHERE color_hist IS NULL",
                [zero_bytes],
            )?;
        }

        Ok(())
    }

    pub fn insert(conn: &Connection, patch: &PatchFeature) -> Result<()> {
        let keypoints_bytes = f32_to_bytes(&patch.keypoints);
        let descriptors_bytes = f32_to_bytes(&patch.descriptors);
        let color_hist_bytes = f32_to_bytes(&patch.color_hist);

        conn.execute(
            r#"
            INSERT OR REPLACE INTO patch_features
            (patch_id, image_id, patch_index, keypoints, descriptors, num_keypoints,
             image_width, image_height, bbox_x, bbox_y, bbox_w, bbox_h, color_hist)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            "#,
            params![
                patch.patch_id,
                patch.image_id,
                patch.patch_index as i32,
                keypoints_bytes,
                descriptors_bytes,
                patch.num_keypoints as i64,
                patch.image_width as i64,
                patch.image_height as i64,
                patch.bbox.x,
                patch.bbox.y,
                patch.bbox.w,
                patch.bbox.h,
                color_hist_bytes,
            ],
        )?;
        Ok(())
    }

    pub fn get_by_image_id(conn: &Connection, image_id: &str) -> Result<Vec<PatchFeature>> {
        let mut stmt = conn.prepare(
            r#"
            SELECT patch_id, image_id, patch_index, keypoints, descriptors,
                   num_keypoints, image_width, image_height, bbox_x, bbox_y, bbox_w, bbox_h,
                   color_hist
            FROM patch_features
            WHERE image_id = ?1
            ORDER BY patch_index
            "#,
        )?;

        let patches = stmt
            .query_map([image_id], |row| {
                let keypoints_blob: Vec<u8> = row.get(3)?;
                let descriptors_blob: Vec<u8> = row.get(4)?;
                let color_hist_blob: Vec<u8> = row.get(12)?;

                Ok(PatchFeature {
                    patch_id: row.get(0)?,
                    image_id: row.get(1)?,
                    patch_index: row.get::<_, i32>(2)? as u8,
                    keypoints: bytes_to_f32(&keypoints_blob),
                    descriptors: bytes_to_f32(&descriptors_blob),
                    num_keypoints: row.get::<_, i64>(5)? as usize,
                    image_width: row.get::<_, i64>(6)? as u32,
                    image_height: row.get::<_, i64>(7)? as u32,
                    bbox: Bbox::new(
                        row.get(8)?,
                        row.get(9)?,
                        row.get(10)?,
                        row.get(11)?,
                    ),
                    color_hist: bytes_to_f32(&color_hist_blob),
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(patches)
    }

    pub fn get_by_patch_id(conn: &Connection, patch_id: &str) -> Result<Option<PatchFeature>> {
        let mut stmt = conn.prepare(
            r#"
            SELECT patch_id, image_id, patch_index, keypoints, descriptors,
                   num_keypoints, image_width, image_height, bbox_x, bbox_y, bbox_w, bbox_h,
                   color_hist
            FROM patch_features
            WHERE patch_id = ?1
            "#,
        )?;

        let mut patches = stmt
            .query_map([patch_id], |row| {
                let keypoints_blob: Vec<u8> = row.get(3)?;
                let descriptors_blob: Vec<u8> = row.get(4)?;
                let color_hist_blob: Vec<u8> = row.get(12)?;

                Ok(PatchFeature {
                    patch_id: row.get(0)?,
                    image_id: row.get(1)?,
                    patch_index: row.get::<_, i32>(2)? as u8,
                    keypoints: bytes_to_f32(&keypoints_blob),
                    descriptors: bytes_to_f32(&descriptors_blob),
                    num_keypoints: row.get::<_, i64>(5)? as usize,
                    image_width: row.get::<_, i64>(6)? as u32,
                    image_height: row.get::<_, i64>(7)? as u32,
                    bbox: Bbox::new(
                        row.get(8)?,
                        row.get(9)?,
                        row.get(10)?,
                        row.get(11)?,
                    ),
                    color_hist: bytes_to_f32(&color_hist_blob),
                })
            })?
            .filter_map(|r| r.ok());

        Ok(patches.next())
    }

    pub fn delete_by_image_id(conn: &Connection, image_id: &str) -> Result<()> {
        conn.execute(
            "DELETE FROM patch_features WHERE image_id = ?1",
            [image_id],
        )?;
        Ok(())
    }

    pub fn count(conn: &Mutex<Connection>) -> Result<i64> {
        let conn = conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM patch_features", [], |row| row.get(0))
    }
}

/// Store aggregated vectors for HNSW indexing
pub struct PatchVectorsTable;

impl PatchVectorsTable {
    pub const TABLE_NAME: &'static str = "patch_vectors";

    pub fn create(conn: &Connection) -> Result<()> {
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS patch_vectors (
                patch_id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                patch_index INTEGER NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY (patch_id) REFERENCES patch_features(patch_id)
            )
            "#,
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_vectors_image_id ON patch_vectors(image_id)",
            [],
        )?;

        Ok(())
    }

    pub fn insert(conn: &Connection, patch_vector: &PatchVector) -> Result<()> {
        let vector_bytes = f32_to_bytes(&patch_vector.vector);

        conn.execute(
            r#"
            INSERT OR REPLACE INTO patch_vectors
            (patch_id, image_id, patch_index, vector)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            params![
                patch_vector.patch_id,
                patch_vector.image_id,
                patch_vector.patch_index as i32,
                vector_bytes,
            ],
        )?;
        Ok(())
    }

    pub fn insert_batch(conn: &Connection, patch_vectors: &[PatchVector]) -> Result<()> {
        let tx = conn.unchecked_transaction()?;

        for pv in patch_vectors {
            let vector_bytes = f32_to_bytes(&pv.vector);

            tx.execute(
                r#"
                INSERT OR REPLACE INTO patch_vectors
                (patch_id, image_id, patch_index, vector)
                VALUES (?1, ?2, ?3, ?4)
                "#,
                params![
                    pv.patch_id,
                    pv.image_id,
                    pv.patch_index as i32,
                    vector_bytes,
                ],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn get_all(conn: &Connection) -> Result<Vec<PatchVector>> {
        let mut stmt = conn.prepare(
            "SELECT patch_id, image_id, patch_index, vector FROM patch_vectors",
        )?;

        let vectors = stmt
            .query_map([], |row| {
                let vector_blob: Vec<u8> = row.get(3)?;
                Ok(PatchVector {
                    patch_id: row.get(0)?,
                    image_id: row.get(1)?,
                    patch_index: row.get::<_, i32>(2)? as u8,
                    vector: bytes_to_f32(&vector_blob),
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(vectors)
    }

    pub fn delete_by_image_id(conn: &Connection, image_id: &str) -> Result<()> {
        conn.execute(
            "DELETE FROM patch_vectors WHERE image_id = ?1",
            [image_id],
        )?;
        Ok(())
    }

    pub fn count(conn: &Connection) -> Result<i64> {
        conn.query_row("SELECT COUNT(*) FROM patch_vectors", [], |row| row.get(0))
    }
}

// Helper: convert f32 Vec to bytes
pub fn f32_to_bytes(vec: &[f32]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4).to_vec()
    }
}

// Helper: convert bytes to f32 Vec
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    let len = bytes.len() / 4;
    let mut vec = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const f32, vec.as_mut_ptr(), len);
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_conversion() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = f32_to_bytes(&original);
        let restored = bytes_to_f32(&bytes);
        assert_eq!(original, restored);
    }
}
