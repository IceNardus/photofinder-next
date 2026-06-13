use flume::{Sender, Receiver, bounded};
use std::sync::Arc;
use anyhow::Result;

use crate::core::database::{Database, tasks::TasksTable};

pub struct Dispatcher {
    db: Arc<Database>,
    image_tx: Sender<DispatchTask>,
}

impl Dispatcher {
    pub fn new(db: Arc<Database>) -> Self {
        let (image_tx, _) = bounded(1000);

        Self {
            db,
            image_tx,
        }
    }

    pub async fn dispatch(&self, image_id: i64) -> Result<()> {
        let task = DispatchTask { image_id };
        self.image_tx.send_async(task).await?;
        Ok(())
    }

    pub async fn dispatch_batch(&self, image_ids: &[i64]) -> Result<usize> {
        let mut dispatched = 0;
        for &image_id in image_ids {
            if self.dispatch(image_id).await.is_ok() {
                dispatched += 1;
            }
        }
        Ok(dispatched)
    }

    pub fn image_receiver(&self) -> Receiver<DispatchTask> {
        let (tx, rx) = bounded(1000);
        rx
    }
}

pub struct DispatchTask {
    pub image_id: i64,
}