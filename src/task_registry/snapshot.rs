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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[derive(Clone, Debug, PartialEq)]
    enum St {
        Running,
        Done,
    }
    impl TaskStatus for St {
        fn is_terminal(&self) -> bool {
            matches!(self, Self::Done)
        }
        fn is_wake_worthy(&self) -> bool {
            matches!(self, Self::Done)
        }
    }

    #[derive(Clone, Debug)]
    struct E {
        id: String,
        status: St,
    }
    impl TaskEntry for E {
        type Status = St;
        fn id(&self) -> &str {
            &self.id
        }
        fn status(&self) -> &St {
            &self.status
        }
        fn set_status(&mut self, s: St) {
            self.status = s;
        }
    }

    fn snap(status: St, finished: bool) -> Snapshot<E> {
        Snapshot {
            id: "t".into(),
            entry: E {
                id: "t".into(),
                status,
            },
            created_at: Instant::now() - Duration::from_millis(100),
            finished_at: if finished { Some(Instant::now()) } else { None },
        }
    }

    #[test]
    fn is_terminal_reflects_entry() {
        assert!(!snap(St::Running, false).is_terminal());
        assert!(snap(St::Done, true).is_terminal());
    }

    #[test]
    fn elapsed_is_positive() {
        let s = snap(St::Running, false);
        assert!(s.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn duration_none_when_running() {
        assert!(snap(St::Running, false).duration().is_none());
    }

    #[test]
    fn duration_some_when_finished() {
        let s = snap(St::Done, true);
        let d = s.duration().unwrap();
        assert!(d >= Duration::ZERO);
        assert!(d <= Duration::from_millis(200));
    }
}
