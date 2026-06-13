use rusqlite::{Connection, params};
use std::sync::Mutex;
use anyhow::Result;

pub struct ImagesTable;

#[derive(Debug, Clone)]
pub struct ImageRecord {
    pub id: i64,
    pub path: String,
    pub hash: String,
    pub size: i64,
    pub modified_time: i64,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub thumbnail_path: Option<String>,
    pub thumbnail_status: String,
    pub scan_status: String,
    pub created_at: i64,
    pub updated_at: i64,
}

pub struct ImageBatchItem {
    pub path: String,
    pub hash: String,
    pub size: i64,
    pub modified_time: i64,
    pub width: Option<i32>,
    pub height: Option<i32>,
}

impl ImagesTable {
    pub fn add(
        conn: &Mutex<Connection>,
        path: &str,
        hash: &str,
        size: i64,
        modified_time: i64,
        width: Option<i32>,
        height: Option<i32>,
    ) -> Result<i64> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO images (path, hash, size, modified_time, width, height, thumbnail_status, scan_status, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', 'pending', ?7, ?8)",
            params![path, hash, size, modified_time, width, height, now, now],
        )?;

        Ok(conn.last_insert_rowid())
    }

    pub fn add_batch(conn: &Mutex<Connection>, items: &[ImageBatchItem]) -> Result<Vec<i64>> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let mut ids = Vec::with_capacity(items.len());
        for item in items {
            conn.execute(
                "INSERT INTO images (path, hash, size, modified_time, width, height, thumbnail_status, scan_status, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', 'pending', ?7, ?8)",
                params![item.path, item.hash, item.size, item.modified_time, item.width, item.height, now, now],
            )?;
            ids.push(conn.last_insert_rowid());
        }
        Ok(ids)
    }

    pub fn get_by_path(conn: &Mutex<Connection>, path: &str) -> Result<Option<ImageRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, hash, size, modified_time, width, height, thumbnail_path, thumbnail_status, scan_status, created_at, updated_at FROM images WHERE path = ?1"
        )?;
        let mut rows = stmt.query(params![path])?;

        if let Some(row) = rows.next()? {
            Ok(Some(ImageRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                modified_time: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                thumbnail_path: row.get(7)?,
                thumbnail_status: row.get(8)?,
                scan_status: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn check_file_signature(
        conn: &Mutex<Connection>,
        path: &str,
        size: i64,
        modified_time: i64,
    ) -> Result<Option<String>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT hash FROM images WHERE path = ?1 AND size = ?2 AND modified_time = ?3"
        )?;
        let mut rows = stmt.query(params![path, size, modified_time])?;

        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn exists_by_hash(conn: &Mutex<Connection>, hash: &str) -> Result<bool> {
        let conn = conn.lock().unwrap();
        let count: i32 = conn.query_row(
            "SELECT COUNT(*) FROM images WHERE hash = ?1",
            params![hash],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    pub fn update_hash(conn: &Mutex<Connection>, path: &str, hash: &str, size: i64, modified_time: i64) -> Result<()> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "UPDATE images SET hash = ?1, size = ?2, modified_time = ?3, updated_at = ?4 WHERE path = ?5",
            params![hash, size, modified_time, now, path],
        )?;
        Ok(())
    }

    pub fn update_scan_status(conn: &Mutex<Connection>, image_id: i64, status: &str) -> Result<()> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "UPDATE images SET scan_status = ?1 WHERE id = ?2",
            params![status, image_id],
        )?;
        Ok(())
    }

    pub fn update_scan_status_batch(conn: &Mutex<Connection>, image_ids: &[i64], status: &str) -> Result<()> {
        let conn = conn.lock().unwrap();
        for id in image_ids {
            conn.execute(
                "UPDATE images SET scan_status = ?1 WHERE id = ?2",
                params![status, id],
            )?;
        }
        Ok(())
    }

    pub fn get_by_hash(conn: &Mutex<Connection>, hash: &str) -> Result<Vec<ImageRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, hash, size, modified_time, width, height, thumbnail_path, thumbnail_status, scan_status, created_at, updated_at FROM images WHERE hash = ?1"
        )?;
        let rows = stmt.query_map(params![hash], |row| {
            Ok(ImageRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                modified_time: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                thumbnail_path: row.get(7)?,
                thumbnail_status: row.get(8)?,
                scan_status: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_pending_thumbnails(conn: &Mutex<Connection>, limit: usize) -> Result<Vec<ImageRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, hash, size, modified_time, width, height, thumbnail_path, thumbnail_status, scan_status, created_at, updated_at FROM images WHERE thumbnail_status = 'pending' LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(ImageRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                modified_time: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                thumbnail_path: row.get(7)?,
                thumbnail_status: row.get(8)?,
                scan_status: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_thumbnail(conn: &Mutex<Connection>, hash: &str, thumbnail_path: &str, status: &str) -> Result<()> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "UPDATE images SET thumbnail_path = ?1, thumbnail_status = ?2 WHERE hash = ?3",
            params![thumbnail_path, status, hash],
        )?;
        Ok(())
    }

    pub fn get_by_id(conn: &Mutex<Connection>, id: i64) -> Result<Option<ImageRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, hash, size, modified_time, width, height, thumbnail_path, thumbnail_status, scan_status, created_at, updated_at FROM images WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            Ok(Some(ImageRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                modified_time: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                thumbnail_path: row.get(7)?,
                thumbnail_status: row.get(8)?,
                scan_status: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn count(conn: &Mutex<Connection>) -> Result<i64> {
        let conn = conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM images", [], |row| row.get(0)).map_err(Into::into)
    }

    pub fn thumbnail_count(conn: &Mutex<Connection>) -> Result<i64> {
        let conn = conn.lock().unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM images WHERE thumbnail_status = 'generated'",
            [],
            |row| row.get(0)
        ).map_err(Into::into)
    }

    pub fn get_all(conn: &Mutex<Connection>, limit: usize, offset: usize) -> Result<Vec<ImageRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, hash, size, modified_time, width, height, thumbnail_path, thumbnail_status, scan_status, created_at, updated_at FROM images LIMIT ?1 OFFSET ?2"
        )?;
        let rows = stmt.query_map(params![limit as i64, offset as i64], |row| {
            Ok(ImageRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                modified_time: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                thumbnail_path: row.get(7)?,
                thumbnail_status: row.get(8)?,
                scan_status: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_all_for_thumbnails(conn: &Mutex<Connection>) -> Result<Vec<ImageRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, hash, size, modified_time, width, height, thumbnail_path, thumbnail_status, scan_status, created_at, updated_at FROM images"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ImageRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                modified_time: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                thumbnail_path: row.get(7)?,
                thumbnail_status: row.get(8)?,
                scan_status: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_pending_tasks(conn: &Mutex<Connection>, limit: usize) -> Result<Vec<ImageRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, hash, size, modified_time, width, height, thumbnail_path, thumbnail_status, scan_status, created_at, updated_at FROM images WHERE scan_status = 'pending' LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(ImageRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                hash: row.get(2)?,
                size: row.get(3)?,
                modified_time: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                thumbnail_path: row.get(7)?,
                thumbnail_status: row.get(8)?,
                scan_status: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
}