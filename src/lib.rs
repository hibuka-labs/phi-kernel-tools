//! Kernel tools for the phi-agent framework.
//!
//! This crate provides the Tool implementations that the LLM uses to interact
//! with agent-works infrastructure:
//!
//! - **file**: read_file, write_file, list_files
//! - **multi-agent**: spawn, send_message, followup_task, wait, list, close
//! - **shell**: execute_command
//!
//! Each tool implements `agent_base::Tool` (or `TypedTool`) and delegates to
//! the corresponding `agent_works` infrastructure types.
//!
//! # Feature gates
//!
//! | Feature | Tools provided |
//! |---------|---------------|
//! | `file` | ReadFileTool, WriteFileTool, ListFilesTool |
//! | `multi-agent` | 6 multi-agent tools |
//! | `shell` | LocalShellTool |
//!
//! All features are opt-in. Use `full` to enable all.

#[cfg(feature = "file")]
pub mod file;

#[cfg(feature = "multi-agent")]
pub mod multi_agent;

#[cfg(feature = "shell")]
pub mod local_shell;
