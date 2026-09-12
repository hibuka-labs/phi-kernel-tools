//! Core traits for the task registry framework.

use std::fmt::Debug;

/// Task status abstraction.
///
/// Business layers implement this to define their own status types.
/// The framework only cares about two properties:
/// - `is_terminal()`: Whether the status is final (no further transitions)
/// - `is_wake_worthy()`: Whether the agent should be notified
///
/// # Example
///
/// ```rust
/// use phi_kernel_tools::task_registry::TaskStatus;
///
/// #[derive(Clone, Debug)]
/// enum ShellStatus {
///     Running,
///     Done,
///     TimedOut,
///     Cancelled,
///     Error(String),
/// }
///
/// impl TaskStatus for ShellStatus {
///     fn is_terminal(&self) -> bool {
///         !matches!(self, Self::Running)
///     }
///
///     fn is_wake_worthy(&self) -> bool {
///         matches!(self, Self::Done | Self::TimedOut | Self::Error(_))
///     }
/// }
/// ```
pub trait TaskStatus: Clone + Debug + Send + Sync + 'static {
    /// Returns true if the status is terminal (no further transitions allowed).
    fn is_terminal(&self) -> bool;

    /// Returns true if the agent should be notified when reaching this status.
    ///
    /// Typically: success/warning states are wake-worthy, cancellation is not.
    fn is_wake_worthy(&self) -> bool;
}

/// Task entry abstraction.
///
/// Business layers implement this for each task type (shell, ops, db, etc.).
/// The entry holds task-specific data and a status that implements `TaskStatus`.
///
/// # Example
///
/// ```rust
/// use phi_kernel_tools::task_registry::{TaskStatus, TaskEntry};
///
/// #[derive(Clone, Debug)]
/// enum ShellStatus { Running, Done, TimedOut, Cancelled, Error(String) }
/// impl TaskStatus for ShellStatus {
///     fn is_terminal(&self) -> bool { !matches!(self, Self::Running) }
///     fn is_wake_worthy(&self) -> bool { matches!(self, Self::Done | Self::TimedOut | Self::Error(_)) }
/// }
///
/// #[derive(Clone, Debug)]
/// struct ShellTask {
///     id: String,
///     command: String,
///     status: ShellStatus,
///     stdout: String,
///     stderr: String,
/// }
///
/// impl TaskEntry for ShellTask {
///     type Status = ShellStatus;
///
///     fn id(&self) -> &str {
///         &self.id
///     }
///
///     fn status(&self) -> &ShellStatus {
///         &self.status
///     }
///
///     fn set_status(&mut self, status: ShellStatus) {
///         self.status = status;
///     }
/// }
/// ```
pub trait TaskEntry: Send + Sync + Debug + 'static {
    /// The status type for this task.
    type Status: TaskStatus;

    /// Returns the task's unique identifier.
    fn id(&self) -> &str;

    /// Returns a reference to the current status.
    fn status(&self) -> &Self::Status;

    /// Sets the task's status.
    fn set_status(&mut self, status: Self::Status);
}

/// Inflight work abstraction.
///
/// Used by the agent to track what type of async work is running.
/// Business layers implement this for their specific work types.
///
/// # Example
///
/// ```rust
/// use phi_kernel_tools::task_registry::InflightWork;
///
/// #[derive(Debug)]
/// struct BackgroundTaskTracker {
///     running: usize,
/// }
///
/// impl InflightWork for BackgroundTaskTracker {
///     fn running_count(&self) -> usize {
///         self.running
///     }
///
///     fn description(&self) -> String {
///         if self.running == 1 {
///             "1 background task".to_string()
///         } else {
///             format!("{} background tasks", self.running)
///         }
///     }
/// }
/// ```
pub trait InflightWork: Send + Debug {
    /// Returns the count of running tasks.
    fn running_count(&self) -> usize;

    /// Returns a human-readable description of what's running.
    fn description(&self) -> String;
}
