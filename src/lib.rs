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
//! - **spill**: SpillSink seam (where truncated shell output goes)
//! - **interaction**: interactive-prompt vocabulary + matcher (pure functions)
//! - **term_session**: PTY transport seam (`TermSession` + `LocalPty`)
//! - **pty_task**: PTY background task supervisor (prompt detection loop)
//! - **task_input**: feed a PTY task's terminal (answer policy tiers)
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
//! | `shell` | LocalShellTool, BackgroundTaskRegistry, PTY task surface (task_input/task_output) |
//!
//! All features are opt-in. Use `full` to enable all.

// Module docs live in each module file as inner `//!` comments. Do NOT add an
// outer `///` doc here: rustdoc then merges the two and resolves the inner
// links in the crate-root scope, so a module's own items (`PromptClass`,
// `TermSession`, ...) stop resolving and `-D warnings` fails the doc build.

pub mod task_registry;

pub mod spill;

pub mod interaction;

#[cfg(feature = "shell")]
pub mod term_session;

#[cfg(feature = "shell")]
pub mod pty_task;

#[cfg(feature = "shell")]
pub mod task_input;

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
