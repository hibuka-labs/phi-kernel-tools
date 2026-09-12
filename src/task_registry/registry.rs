//! Generic task registry implementation.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::snapshot::Snapshot;
use super::traits::{TaskEntry, TaskStatus};

/// Error returned when task registration fails.
#[derive(Debug, Clone)]
pub enum RegisterError {
    /// Too many running tasks.
    LimitExceeded,
}

impl std::fmt::Display for RegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LimitExceeded => write!(f, "too many tasks running"),
        }
    }
}

/// Internal slot holding a task entry with lifecycle metadata.
#[derive(Debug)]
pub(crate) struct Slot<E: TaskEntry> {
    pub(crate) entry: E,
    pub(crate) cancel_token: CancellationToken,
    pub(crate) created_at: Instant,
    pub(crate) finished_at: Option<Instant>,
}

/// Generic task registry.
///
/// Manages the lifecycle of async tasks: registration, status updates,
/// cancellation, GC, and watch-based notification.
///
/// Business layers create `TaskRegistry<E>` with their own `TaskEntry`
/// implementation. The registry doesn't know or care what the task does.
///
/// # Thread safety
///
/// `TaskRegistry` is `Send + Sync` and intended to be shared as `Arc<TaskRegistry<E>>`.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use tokio_util::sync::CancellationToken;
/// use phi_kernel_tools::task_registry::{TaskRegistry, TaskEntry, TaskStatus};
///
/// // Business layer defines their task type
/// #[derive(Clone, Debug)]
/// enum MyStatus { Running, Done, Error(String) }
/// impl TaskStatus for MyStatus {
///     fn is_terminal(&self) -> bool { !matches!(self, Self::Running) }
///     fn is_wake_worthy(&self) -> bool { matches!(self, Self::Done | Self::Error(_)) }
/// }
///
/// #[derive(Clone, Debug)]
/// struct MyTask { id: String, status: MyStatus }
/// impl TaskEntry for MyTask {
///     type Status = MyStatus;
///     fn id(&self) -> &str { &self.id }
///     fn status(&self) -> &MyStatus { &self.status }
///     fn set_status(&mut self, status: MyStatus) { self.status = status; }
/// }
///
/// // Create registry
/// let registry = TaskRegistry::<MyTask>::new(4);
///
/// // Register a task
/// let token = CancellationToken::new();
/// let task = MyTask { id: "task_1".into(), status: MyStatus::Running };
/// let task_id = registry.register(task, token).unwrap();
///
/// // Update status
/// registry.update_status(&task_id, MyStatus::Done);
/// ```
#[derive(Debug)]
pub struct TaskRegistry<E: TaskEntry> {
    pub(crate) slots: RwLock<HashMap<String, Slot<E>>>,
    max_tasks: usize,
    status_tx: watch::Sender<()>,
    status_rx: watch::Receiver<()>,
}

impl<E: TaskEntry> TaskRegistry<E> {
    /// Creates a new registry with the given maximum concurrent task limit.
    pub fn new(max_tasks: usize) -> Arc<Self> {
        let (status_tx, status_rx) = watch::channel(());
        Arc::new(Self {
            slots: RwLock::new(HashMap::new()),
            max_tasks,
            status_tx,
            status_rx,
        })
    }

    /// Registers a new task.
    ///
    /// Returns `Ok(task_id)` or `Err(RegisterError::LimitExceeded)` if the
    /// limit is reached.
    pub fn register(&self, entry: E, cancel_token: CancellationToken) -> Result<String, RegisterError> {
        let mut map = self.slots.write().unwrap();
        let running = map
            .values()
            .filter(|s| !s.entry.status().is_terminal())
            .count();

        if running >= self.max_tasks {
            return Err(RegisterError::LimitExceeded);
        }

        let id = entry.id().to_string();
        let slot = Slot {
            entry,
            cancel_token,
            created_at: Instant::now(),
            finished_at: None,
        };
        map.insert(id.clone(), slot);
        let _ = self.status_tx.send(());
        Ok(id)
    }

    /// Updates the status of a task.
    ///
    /// **Terminal-state protection**: if the task is already in a terminal
    /// state, the new status is silently ignored.
    pub fn update_status(&self, id: &str, status: E::Status) {
        let mut map = self.slots.write().unwrap();
        if let Some(slot) = map.get_mut(id) {
            if slot.entry.status().is_terminal() {
                tracing::warn!(
                    task_id = id,
                    current = ?slot.entry.status(),
                    requested = ?status,
                    "update_status: ignoring — already in terminal state"
                );
                return;
            }

            let is_terminal = status.is_terminal();
            slot.entry.set_status(status);
            if is_terminal {
                slot.finished_at = Some(Instant::now());
            }
            let _ = self.status_tx.send(());
        }
    }

    /// Gets a snapshot of a single task.
    pub fn snapshot(&self, id: &str) -> Option<Snapshot<E>>
    where
        E: Clone,
    {
        let map = self.slots.read().unwrap();
        map.get(id).map(|slot| Snapshot {
            id: id.to_string(),
            entry: slot.entry.clone(),
            created_at: slot.created_at,
            finished_at: slot.finished_at,
        })
    }

    /// Gets snapshots of all tasks, and GC entries finished longer than `gc_ttl`.
    ///
    /// Designed to be called from the TUI tick loop (every 250ms).
    pub fn snapshot_all(&self, gc_ttl: Duration) -> Vec<Snapshot<E>>
    where
        E: Clone,
    {
        let mut map = self.slots.write().unwrap();

        // GC: remove tasks that finished longer than gc_ttl ago.
        map.retain(|_, slot| match slot.finished_at {
            Some(finished) => finished.elapsed() < gc_ttl,
            None => true, // still running — keep
        });

        map.iter()
            .map(|(id, slot)| Snapshot {
                id: id.clone(),
                entry: slot.entry.clone(),
                created_at: slot.created_at,
                finished_at: slot.finished_at,
            })
            .collect()
    }

    /// Cancels a task.
    ///
    /// Triggers the `CancellationToken` and sets the status to terminal.
    /// Returns `true` if the task was found and cancelled, `false` otherwise.
    pub fn cancel(&self, id: &str) -> bool {
        let mut map = self.slots.write().unwrap();
        if let Some(slot) = map.get_mut(id) {
            if slot.entry.status().is_terminal() {
                return false;
            }
            slot.cancel_token.cancel();
            true
        } else {
            false
        }
    }

    /// Checks if a task's cancel token has been triggered.
    pub fn is_cancelled(&self, id: &str) -> bool {
        let map = self.slots.read().unwrap();
        map.get(id)
            .map(|slot| slot.cancel_token.is_cancelled())
            .unwrap_or(false)
    }

    /// Waits for a task to reach a terminal state.
    ///
    /// Returns `Some(snapshot)` when the task finishes, or `None` on timeout
    /// or if the task doesn't exist.
    pub async fn wait(&self, id: &str, timeout: Duration) -> Option<Snapshot<E>>
    where
        E: Clone,
    {
        // Quick check: already terminal?
        if let Some(snap) = self.snapshot(id) {
            if snap.entry.status().is_terminal() {
                return Some(snap);
            }
        } else {
            return None; // task not found
        }

        let mut rx = self.status_rx.clone();
        let deadline = Instant::now() + timeout;

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return self.snapshot(id);
            }

            match tokio::time::timeout(remaining, rx.changed()).await {
                Ok(Ok(())) => {
                    if let Some(snap) = self.snapshot(id) {
                        if snap.entry.status().is_terminal() {
                            return Some(snap);
                        }
                    } else {
                        return None; // GC'd while waiting
                    }
                }
                Ok(Err(_)) => return self.snapshot(id), // sender dropped
                Err(_) => return self.snapshot(id),      // timeout
            }
        }
    }

    /// Returns a `watch::Receiver` for detecting status changes.
    pub fn watch(&self) -> watch::Receiver<()> {
        self.status_rx.clone()
    }

    /// Shutdown: cancel all running tasks and clear the registry.
    pub fn shutdown(&self) {
        let mut map = self.slots.write().unwrap();
        for (_, slot) in map.iter_mut() {
            if !slot.entry.status().is_terminal() {
                slot.cancel_token.cancel();
            }
        }
        map.clear();
        let _ = self.status_tx.send(());
    }

    /// Returns the count of running (non-terminal) tasks.
    pub fn running_count(&self) -> usize {
        let map = self.slots.read().unwrap();
        map.values()
            .filter(|s| !s.entry.status().is_terminal())
            .count()
    }

    /// Provides mutable access to a task entry for business-layer mutations.
    ///
    /// This allows business layers to modify task-specific fields (output buffers,
    /// metadata, etc.) while the framework handles lifecycle management.
    ///
    /// Returns `None` if the task doesn't exist.
    pub fn with_entry_mut<F, R>(&self, id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut E) -> R,
    {
        let mut map = self.slots.write().unwrap();
        map.get_mut(id).map(|slot| f(&mut slot.entry))
    }

    /// Provides read access to a task entry.
    ///
    /// Returns `None` if the task doesn't exist.
    pub fn with_entry<F, R>(&self, id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&E) -> R,
    {
        let map = self.slots.read().unwrap();
        map.get(id).map(|slot| f(&slot.entry))
    }
}
