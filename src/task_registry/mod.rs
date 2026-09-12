//! Generic task registry framework.
//!
//! This module provides a business-agnostic framework for managing async tasks.
//! It decouples task lifecycle management from task-specific content, enabling
//! reuse across different agent types (shell, ops, db, etc.).
//!
//! # Architecture
//!
//! - **Framework layer** (this module): Lifecycle management, terminal-state
//!   protection, watch-based notification, GC
//! - **Business layer**: Task status definitions, task entry implementations,
//!   output buffering, result formatting
//! - **TUI layer**: Rendering, snapshot consumption, wake injection
//!
//! # Key abstractions
//!
//! - `TaskStatus` trait: Defines terminal/wake-worthy semantics
//! - `TaskEntry` trait: Business layer implements this for each task type
//! - `TaskRegistry<E>`: Generic registry managing any `E: TaskEntry`

mod traits;
mod registry;
mod snapshot;

pub use traits::{TaskStatus, TaskEntry, InflightWork};
pub use registry::TaskRegistry;
pub use snapshot::Snapshot;

// Re-export commonly used types
pub use tokio_util::sync::CancellationToken;
