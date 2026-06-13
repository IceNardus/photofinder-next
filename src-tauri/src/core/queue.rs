use std::sync::Arc;
use flume::{Sender, Receiver, bounded};
use anyhow::Result;

use crate::core::database::{Database, tasks::{TasksTable, TaskType, TaskStatus}};

pub struct TaskQueue {
    db: Arc<Database>,
    image_tx: Sender<QueuedTask>,
}

impl TaskQueue {
    pub fn new(db: Arc<Database>) -> Self {
        let (image_tx, _) = bounded(1000);

        Self {
            db,
            image_tx,
        }
    }

    pub async fn enqueue(&self, image_id: i64) -> Result<()> {
        let task = QueuedTask { image_id };
        self.image_tx.send_async(task).await?;
        Ok(())
    }

    pub fn get_pending_image_tasks(&self, limit: usize) -> Result<Vec<QueuedTask>> {
        let tasks = TasksTable::get_pending(&self.db.conn, limit)?;
        Ok(tasks.into_iter().map(|t| QueuedTask {
            image_id: t.image_id,
        }).collect())
    }

    pub fn mark_completed(&self, task_id: i64) -> Result<()> {
        TasksTable::update_status(&self.db.conn, task_id, TaskStatus::Completed)?;
        Ok(())
    }

    pub fn mark_failed(&self, task_id: i64) -> Result<()> {
        TasksTable::increment_retry(&self.db.conn, task_id)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QueuedTask {
    pub image_id: i64,
}