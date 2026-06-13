use rusqlite::{Connection, params};
use std::sync::Mutex;
use anyhow::Result;

pub struct FacesTable;

#[derive(Debug, Clone)]
pub struct FaceRecord {
    pub id: i64,
    pub image_id: i64,
    pub person_id: Option<i64>,
    pub bbox: [f32; 4],
    pub detector_score: f32,
    pub blur_score: f32,
    pub pose_score: f32,
    pub face_area_score: f32,
    pub quality: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub vector_offset: i64,
    pub vector_length: i32,
    pub storage_status: String,
    pub index_status: String,
}

#[derive(Debug, Clone)]
pub struct FaceQualityBreakdown {
    pub detector_score: f32,
    pub blur_score: f32,
    pub pose_score: f32,
    pub face_area_score: f32,
    pub final_score: f32,
}

impl FacesTable {
    pub fn get_by_id(conn: &Mutex<Connection>, id: i64) -> Result<Option<FaceRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detector_score, blur_score, pose_score, face_area_score, quality, yaw, pitch, roll, vector_offset, vector_length, storage_status, index_status FROM faces WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            Ok(Some(FaceRecord {
                id: row.get(0)?,
                image_id: row.get(1)?,
                person_id: row.get(2)?,
                bbox: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
                detector_score: row.get(7)?,
                blur_score: row.get(8)?,
                pose_score: row.get(9)?,
                face_area_score: row.get(10)?,
                quality: row.get(11)?,
                yaw: row.get(12)?,
                pitch: row.get(13)?,
                roll: row.get(14)?,
                vector_offset: row.get(15)?,
                vector_length: row.get(16)?,
                storage_status: row.get(17)?,
                index_status: row.get(18)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn add(
        conn: &Mutex<Connection>,
        image_id: i64,
        bbox: &[f32; 4],
        detector_score: f32,
        blur_score: f32,
        pose_score: f32,
        face_area_score: f32,
        quality: f32,
        yaw: f32,
        pitch: f32,
        roll: f32,
        vector_offset: i64,
        vector_length: i32,
    ) -> Result<i64> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "INSERT INTO faces (image_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detector_score, blur_score, pose_score, face_area_score, quality, yaw, pitch, roll, vector_offset, vector_length, storage_status, index_status) VALUES (?1, NULL, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, 'pending', 'pending')",
            params![image_id, bbox[0], bbox[1], bbox[2], bbox[3], detector_score, blur_score, pose_score, face_area_score, quality, yaw, pitch, roll, vector_offset, vector_length],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn count(conn: &Mutex<Connection>) -> Result<i64> {
        let conn = conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM faces", [], |row| row.get(0)).map_err(Into::into)
    }

    pub fn get_by_image(conn: &Mutex<Connection>, image_id: i64) -> Result<Vec<FaceRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detector_score, blur_score, pose_score, face_area_score, quality, yaw, pitch, roll, vector_offset, vector_length, storage_status, index_status FROM faces WHERE image_id = ?1"
        )?;
        let rows = stmt.query_map(params![image_id], |row| {
            Ok(FaceRecord {
                id: row.get(0)?,
                image_id: row.get(1)?,
                person_id: row.get(2)?,
                bbox: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
                detector_score: row.get(7)?,
                blur_score: row.get(8)?,
                pose_score: row.get(9)?,
                face_area_score: row.get(10)?,
                quality: row.get(11)?,
                yaw: row.get(12)?,
                pitch: row.get(13)?,
                roll: row.get(14)?,
                vector_offset: row.get(15)?,
                vector_length: row.get(16)?,
                storage_status: row.get(17)?,
                index_status: row.get(18)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_pending_storage(conn: &Mutex<Connection>, limit: usize) -> Result<Vec<FaceRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detector_score, blur_score, pose_score, face_area_score, quality, yaw, pitch, roll, vector_offset, vector_length, storage_status, index_status FROM faces WHERE storage_status = 'pending' LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(FaceRecord {
                id: row.get(0)?,
                image_id: row.get(1)?,
                person_id: row.get(2)?,
                bbox: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
                detector_score: row.get(7)?,
                blur_score: row.get(8)?,
                pose_score: row.get(9)?,
                face_area_score: row.get(10)?,
                quality: row.get(11)?,
                yaw: row.get(12)?,
                pitch: row.get(13)?,
                roll: row.get(14)?,
                vector_offset: row.get(15)?,
                vector_length: row.get(16)?,
                storage_status: row.get(17)?,
                index_status: row.get(18)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_pending_index(conn: &Mutex<Connection>, limit: usize) -> Result<Vec<FaceRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detector_score, blur_score, pose_score, face_area_score, quality, yaw, pitch, roll, vector_offset, vector_length, storage_status, index_status FROM faces WHERE storage_status = 'stored' AND index_status = 'pending' LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(FaceRecord {
                id: row.get(0)?,
                image_id: row.get(1)?,
                person_id: row.get(2)?,
                bbox: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
                detector_score: row.get(7)?,
                blur_score: row.get(8)?,
                pose_score: row.get(9)?,
                face_area_score: row.get(10)?,
                quality: row.get(11)?,
                yaw: row.get(12)?,
                pitch: row.get(13)?,
                roll: row.get(14)?,
                vector_offset: row.get(15)?,
                vector_length: row.get(16)?,
                storage_status: row.get(17)?,
                index_status: row.get(18)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_storage_status(conn: &Mutex<Connection>, face_id: i64, status: &str) -> Result<()> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "UPDATE faces SET storage_status = ?1 WHERE id = ?2",
            params![status, face_id],
        )?;
        Ok(())
    }

    pub fn update_index_status(conn: &Mutex<Connection>, face_id: i64, status: &str) -> Result<()> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "UPDATE faces SET index_status = ?1 WHERE id = ?2",
            params![status, face_id],
        )?;
        Ok(())
    }

    pub fn mark_stored(conn: &Mutex<Connection>, face_ids: &[i64]) -> Result<()> {
        let conn = conn.lock().unwrap();
        for id in face_ids {
            conn.execute(
                "UPDATE faces SET storage_status = 'stored' WHERE id = ?1",
                params![id],
            )?;
        }
        Ok(())
    }

    pub fn mark_indexed(conn: &Mutex<Connection>, face_ids: &[i64]) -> Result<()> {
        let conn = conn.lock().unwrap();
        for id in face_ids {
            conn.execute(
                "UPDATE faces SET index_status = 'indexed' WHERE id = ?1",
                params![id],
            )?;
        }
        Ok(())
    }

    pub fn update_person_id(conn: &Mutex<Connection>, face_id: i64, person_id: i64) -> Result<()> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "UPDATE faces SET person_id = ?1 WHERE id = ?2",
            params![person_id, face_id],
        )?;
        Ok(())
    }

    pub fn get_by_person(conn: &Mutex<Connection>, person_id: i64) -> Result<Vec<FaceRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detector_score, blur_score, pose_score, face_area_score, quality, yaw, pitch, roll, vector_offset, vector_length, storage_status, index_status FROM faces WHERE person_id = ?1"
        )?;
        let rows = stmt.query_map(params![person_id], |row| {
            Ok(FaceRecord {
                id: row.get(0)?,
                image_id: row.get(1)?,
                person_id: row.get(2)?,
                bbox: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
                detector_score: row.get(7)?,
                blur_score: row.get(8)?,
                pose_score: row.get(9)?,
                face_area_score: row.get(10)?,
                quality: row.get(11)?,
                yaw: row.get(12)?,
                pitch: row.get(13)?,
                roll: row.get(14)?,
                vector_offset: row.get(15)?,
                vector_length: row.get(16)?,
                storage_status: row.get(17)?,
                index_status: row.get(18)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
}