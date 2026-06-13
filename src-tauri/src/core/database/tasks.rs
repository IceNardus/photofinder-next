use rusqlite::{Connection, params};
use std::sync::Mutex;
use anyhow::Result;

pub struct TasksTable;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskType {
    Image,  // Single task for face and patch processing
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Image => "image",
        }
    }
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::Processing => "processing",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
        }
    }
}

impl TasksTable {
    pub fn add(conn: &Mutex<Connection>, image_id: i64, _task_type: TaskType) -> Result<i64> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO scan_tasks (image_id, task_type, status, retry_count, created_at) VALUES (?1, 'image', 'pending', 0, ?2)",
            params![image_id, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn add_batch(conn: &Mutex<Connection>, image_ids: &[i64]) -> Result<usize> {
        let conn = conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let mut count = 0;
        for image_id in image_ids {
            conn.execute(
                "INSERT INTO scan_tasks (image_id, task_type, status, retry_count, created_at) VALUES (?1, 'image', 'pending', 0, ?2)",
                params![image_id, now],
            )?;
            count += 1;
        }
        Ok(count)
    }

    pub fn get_pending(conn: &Mutex<Connection>, limit: usize) -> Result<Vec<TaskRecord>> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, image_id, task_type, status, retry_count, created_at FROM scan_tasks WHERE task_type = 'image' AND status = 'pending' LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(TaskRecord {
                id: row.get(0)?,
                image_id: row.get(1)?,
                task_type: row.get(2)?,
                status: row.get(3)?,
                retry_count: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_status(conn: &Mutex<Connection>, task_id: i64, status: TaskStatus) -> Result<()> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "UPDATE scan_tasks SET status = ?1 WHERE id = ?2",
            params![status.as_str(), task_id],
        )?;
        Ok(())
    }

    pub fn update_status_batch(conn: &Mutex<Connection>, task_ids: &[i64], status: TaskStatus) -> Result<()> {
        let conn = conn.lock().unwrap();
        let status_str = status.as_str();
        for task_id in task_ids {
            conn.execute(
                "UPDATE scan_tasks SET status = ?1 WHERE id = ?2",
                params![status_str, task_id],
            )?;
        }
        Ok(())
    }

    pub fn increment_retry(conn: &Mutex<Connection>, task_id: i64) -> Result<()> {
        let conn = conn.lock().unwrap();
        conn.execute(
            "UPDATE scan_tasks SET retry_count = retry_count + 1 WHERE id = ?1",
            params![task_id],
        )?;
        Ok(())
    }

    pub fn pending_count(conn: &Mutex<Connection>) -> Result<i64> {
        let conn = conn.lock().unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM scan_tasks WHERE status = 'pending'",
            [],
            |row| row.get(0),
        )
        .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub id: i64,
    pub image_id: i64,
    pub task_type: String,
    pub status: String,
    pub retry_count: i32,
    pub created_at: i64,
}