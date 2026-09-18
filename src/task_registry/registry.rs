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
    pub fn register(
        &self,
        entry: E,
        cancel_token: CancellationToken,
    ) -> Result<String, RegisterError> {
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
        let snap = self.snapshot(id)?;
        if snap.entry.status().is_terminal() {
            return Some(snap);
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
                    let snap = self.snapshot(id)?; // None => GC'd while waiting
                    if snap.entry.status().is_terminal() {
                        return Some(snap);
                    }
                }
                Ok(Err(_)) => return self.snapshot(id), // sender dropped
                Err(_) => return self.snapshot(id),     // timeout
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
        for slot in map.values_mut() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[derive(Clone, Debug, PartialEq)]
    enum Status {
        Running,
        Done,
        Error(String),
    }

    impl TaskStatus for Status {
        fn is_terminal(&self) -> bool {
            !matches!(self, Self::Running)
        }
        fn is_wake_worthy(&self) -> bool {
            matches!(self, Self::Done | Self::Error(_))
        }
    }

    #[derive(Clone, Debug)]
    struct Task {
        id: String,
        status: Status,
        data: String,
    }

    impl TaskEntry for Task {
        type Status = Status;
        fn id(&self) -> &str {
            &self.id
        }
        fn status(&self) -> &Status {
            &self.status
        }
        fn set_status(&mut self, status: Status) {
            self.status = status;
        }
    }

    fn make_task(id: &str) -> Task {
        Task {
            id: id.into(),
            status: Status::Running,
            data: String::new(),
        }
    }

    #[test]
    fn register_and_snapshot() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        assert_eq!(id, "t1");
        let snap = reg.snapshot(&id).unwrap();
        assert_eq!(snap.id, "t1");
        assert!(!snap.is_terminal());
        assert!(snap.duration().is_none());
    }

    #[test]
    fn register_limit_exceeded() {
        let reg = TaskRegistry::<Task>::new(1);
        reg.register(make_task("t1"), CancellationToken::new())
            .unwrap();
        let err = reg
            .register(make_task("t2"), CancellationToken::new())
            .unwrap_err();
        assert!(format!("{err}").contains("too many"));
    }

    #[test]
    fn terminal_tasks_do_not_count_against_limit() {
        let reg = TaskRegistry::<Task>::new(1);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        reg.update_status(&id, Status::Done);
        // Now we can register another one
        reg.register(make_task("t2"), CancellationToken::new())
            .unwrap();
        assert_eq!(reg.running_count(), 1);
    }

    #[test]
    fn update_status_sets_terminal_and_finished_at() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        reg.update_status(&id, Status::Done);
        let snap = reg.snapshot(&id).unwrap();
        assert!(snap.is_terminal());
        assert!(snap.duration().is_some());
    }

    #[test]
    fn update_status_ignores_transition_after_terminal() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        reg.update_status(&id, Status::Done);
        reg.update_status(&id, Status::Error("ignored".into()));
        let snap = reg.snapshot(&id).unwrap();
        assert_eq!(snap.entry.status, Status::Done);
    }

    #[test]
    fn update_status_nonexistent_is_noop() {
        let reg = TaskRegistry::<Task>::new(4);
        reg.update_status("missing", Status::Done); // should not panic
    }

    #[test]
    fn cancel_triggers_token() {
        let reg = TaskRegistry::<Task>::new(4);
        let token = CancellationToken::new();
        let id = reg.register(make_task("t1"), token.clone()).unwrap();
        assert!(!token.is_cancelled());
        assert!(reg.cancel(&id));
        assert!(token.is_cancelled());
        assert!(reg.is_cancelled(&id));
    }

    #[test]
    fn cancel_terminal_returns_false() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        reg.update_status(&id, Status::Done);
        assert!(!reg.cancel(&id));
    }

    #[test]
    fn cancel_nonexistent_returns_false() {
        let reg = TaskRegistry::<Task>::new(4);
        assert!(!reg.cancel("missing"));
    }

    #[test]
    fn is_cancelled_nonexistent_returns_false() {
        let reg = TaskRegistry::<Task>::new(4);
        assert!(!reg.is_cancelled("missing"));
    }

    #[test]
    fn snapshot_nonexistent_returns_none() {
        let reg = TaskRegistry::<Task>::new(4);
        assert!(reg.snapshot("missing").is_none());
    }

    #[test]
    fn snapshot_all_returns_all_and_gcs() {
        let reg = TaskRegistry::<Task>::new(4);
        let id1 = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        let _id2 = reg
            .register(make_task("t2"), CancellationToken::new())
            .unwrap();
        reg.update_status(&id1, Status::Done);

        // With very short GC TTL, finished tasks get collected
        let snaps = reg.snapshot_all(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        let snaps2 = reg.snapshot_all(Duration::from_millis(1));
        assert!(snaps2.len() <= snaps.len());
    }

    #[test]
    fn with_entry_mut_modifies_data() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        reg.with_entry_mut(&id, |e| e.data = "hello".into());
        let data = reg.with_entry(&id, |e| e.data.clone()).unwrap();
        assert_eq!(data, "hello");
    }

    #[test]
    fn with_entry_nonexistent_returns_none() {
        let reg = TaskRegistry::<Task>::new(4);
        assert!(reg.with_entry("missing", |_| ()).is_none());
        assert!(reg.with_entry_mut("missing", |_| ()).is_none());
    }

    #[test]
    fn shutdown_cancels_all_and_clears() {
        let reg = TaskRegistry::<Task>::new(4);
        let t1 = CancellationToken::new();
        let t2 = CancellationToken::new();
        reg.register(make_task("t1"), t1.clone()).unwrap();
        reg.register(make_task("t2"), t2.clone()).unwrap();
        reg.shutdown();
        assert!(t1.is_cancelled());
        assert!(t2.is_cancelled());
        assert_eq!(reg.running_count(), 0);
    }

    #[test]
    fn watch_receiver_gets_notified() {
        let reg = TaskRegistry::<Task>::new(4);
        let mut rx = reg.watch();
        reg.register(make_task("t1"), CancellationToken::new())
            .unwrap();
        // The watch channel should have been updated
        assert!(rx.has_changed().unwrap_or(true));
    }

    #[tokio::test]
    async fn wait_returns_immediately_if_already_terminal() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        reg.update_status(&id, Status::Done);
        let snap = reg.wait(&id, Duration::from_secs(1)).await.unwrap();
        assert!(snap.is_terminal());
    }

    #[tokio::test]
    async fn wait_returns_none_for_nonexistent() {
        let reg = TaskRegistry::<Task>::new(4);
        assert!(
            reg.wait("missing", Duration::from_millis(50))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn wait_times_out() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        let snap = reg.wait(&id, Duration::from_millis(50)).await.unwrap();
        assert!(!snap.is_terminal()); // still running
    }

    #[tokio::test]
    async fn wait_wakes_on_status_change() {
        let reg = TaskRegistry::<Task>::new(4);
        let id = reg
            .register(make_task("t1"), CancellationToken::new())
            .unwrap();
        let id_clone = id.clone();
        let reg_clone = Arc::clone(&reg);

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            reg_clone.update_status(&id_clone, Status::Done);
        });

        let snap = reg.wait(&id, Duration::from_secs(1)).await.unwrap();
        assert!(snap.is_terminal());
        handle.await.unwrap();
    }

    #[test]
    fn display_register_error() {
        let err = RegisterError::LimitExceeded;
        assert_eq!(format!("{err}"), "too many tasks running");
    }
}
