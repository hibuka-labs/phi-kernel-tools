//! Snapshot representation for task queries.

use std::time::Instant;

use super::traits::{TaskEntry, TaskStatus};

/// Task snapshot for read-only queries.
///
/// Provides a view of a task's state without holding the registry lock.
/// Business layers can extend this via `TaskEntry` to include task-specific
/// data (command, output buffers, exit code, etc.).
#[derive(Debug, Clone)]
pub struct Snapshot<E: TaskEntry + Clone> {
    /// Task identifier.
    pub id: String,
    /// Cloned task entry (contains status and task-specific data).
    pub entry: E,
    /// When the task was registered.
    pub created_at: Instant,
    /// When the task reached a terminal state (if it has).
    pub finished_at: Option<Instant>,
}

impl<E: TaskEntry + Clone> Snapshot<E> {
    /// Returns whether the task is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.entry.status().is_terminal()
    }

    /// Returns the elapsed time since task creation.
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns the duration the task was running (if finished).
    pub fn duration(&self) -> Option<std::time::Duration> {
        self.finished_at.map(|finished| finished - self.created_at)
    }
}
