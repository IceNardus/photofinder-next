use std::sync::Arc;
use rayon::ThreadPool;
use anyhow::Result;
use std::thread;
use num_cpus;

pub struct WorkerPool {
    thread_pool: ThreadPool,
    face_worker_count: usize,
}

impl WorkerPool {
    pub fn new() -> Self {
        let cpu_count = num_cpus::get();
        let worker_count = std::cmp::max(2, cpu_count - 2);

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_count)
            .build()
            .unwrap();

        Self {
            thread_pool,
            face_worker_count: worker_count,
        }
    }

    pub fn face_worker_count(&self) -> usize {
        self.face_worker_count
    }

    pub fn total_worker_count(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    pub fn execute_parallel<F, R>(&self, tasks: Vec<F>) -> Vec<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        use rayon::prelude::*;
        tasks.into_par_iter().map(|t| t()).collect()
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}