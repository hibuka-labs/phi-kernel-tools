//! Kernel tools for the phi-agent framework.
//!
//! This crate provides the Tool implementations that the LLM uses to interact
//! with agent-works infrastructure:
//!
//! - **file**: read_file, write_file, list_files
//! - **context_rotation**: notes, history, handoff (context window management)
//! - **multi-agent**: spawn, send_message, followup_task, wait, list, close
//! - **shell**: execute_command
//! - **task_registry**: Generic task registry framework (business-agnostic)
//!
//! Each tool implements `agent_base::Tool` (or `TypedTool`) and delegates to
//! the corresponding `agent_works` infrastructure types.
//!
//! # Feature gates
//!
//! | Feature | Tools provided |
//! |---------|---------------|
//! | `file` | ReadFileTool, WriteFileTool, ListFilesTool |
//! | `context_rotation` | NotesStore, HistoryStore, extract_ledger + 9 LLM tools |
//! | `multi-agent` | 6 multi-agent tools |
//! | `shell` | LocalShellTool, BackgroundTaskRegistry |
//!
//! All features are opt-in. Use `full` to enable all.

/// Generic task registry framework (always available).
pub mod task_registry;

#[cfg(feature = "file")]
pub mod file;

#[cfg(feature = "context")]
pub mod context_rotation;

#[cfg(feature = "multi-agent")]
pub mod multi_agent;

#[cfg(feature = "shell")]
pub mod local_shell;

#[cfg(feature = "shell")]
pub mod background_shell;
