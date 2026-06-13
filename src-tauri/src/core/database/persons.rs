use rusqlite::{Connection, params};
use std::sync::Mutex;
use anyhow::Result;

pub struct PersonsTable;

#[derive(Debug, Clone)]
pub struct PersonRecord {
    pub id: i64,
    pub face_count: i64,
    pub cover_face_id: Option<i64>,
    pub center_vector_offset: i64,
    pub center_vector_length: i32,
    pub created_at: i64,
    pub updated_at: i64,
}

impl PersonsTable {
    pub fn create(conn: &Mutex<Connection>) -> Result<()> {
        let conn = conn.lock().unwrap();
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_face_count ON persons(face_count)",
            [],
        )?;
        Ok(())
    }

    pub fn add(
        conn: &Mutex<Connection>,
        face_count: i64,
        cover_face_id: Option<i64>,
        center_vector_offset: i64,
        center_vector_length: i32,
    ) -> Result<i64> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO persons (face_count, cover_face_id, center_vector_offset, center_vector_length, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![face_count, cover_face_id, center_vector_offset, center_vector_length, now, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_by_id(conn: &Mutex<Connection>, id: i64) -> Result<Option<PersonRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, face_count, cover_face_id, center_vector_offset, center_vector_length, created_at, updated_at FROM persons WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            Ok(Some(PersonRecord {
                id: row.get(0)?,
                face_count: row.get(1)?,
                cover_face_id: row.get(2)?,
                center_vector_offset: row.get(3)?,
                center_vector_length: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn increment_face_count(conn: &Mutex<Connection>, person_id: i64) -> Result<()> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "UPDATE persons SET face_count = face_count + 1, updated_at = ?1 WHERE id = ?2",
            params![now, person_id],
        )?;
        Ok(())
    }

    pub fn update_cover_face(conn: &Mutex<Connection>, person_id: i64, face_id: i64) -> Result<()> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "UPDATE persons SET cover_face_id = ?1, updated_at = ?2 WHERE id = ?3",
            params![face_id, now, person_id],
        )?;
        Ok(())
    }

    pub fn update_center_vector(conn: &Mutex<Connection>, person_id: i64, offset: i64, length: i32) -> Result<()> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "UPDATE persons SET center_vector_offset = ?1, center_vector_length = ?2, updated_at = ?3 WHERE id = ?4",
            params![offset, length, now, person_id],
        )?;
        Ok(())
    }

    pub fn count(conn: &Mutex<Connection>) -> Result<i64> {
        let conn = conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM persons", [], |row| row.get(0)).map_err(Into::into)
    }

    pub fn get_all(conn: &Mutex<Connection>) -> Result<Vec<PersonRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, face_count, cover_face_id, center_vector_offset, center_vector_length, created_at, updated_at FROM persons ORDER BY face_count DESC"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(PersonRecord {
                id: row.get(0)?,
                face_count: row.get(1)?,
                cover_face_id: row.get(2)?,
                center_vector_offset: row.get(3)?,
                center_vector_length: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
}