//! Context rotation — tools and I/O for managing context window lifecycle.
//!
//! This module provides the tools that the LLM uses to persist state across
//! context window rotations:
//!
//! - **notes**: persistent scratchpad (`notes_read_file`, `notes_write_file`,
//!   `notes_list_files`, `notes_search_contents`, `notes_append_to_file`)
//! - **history**: archived window access (`history_list_windows`,
//!   `history_list_items`, `history_read_item`, `history_search_contents`)
//! - **handoff**: mechanical activity ledger, plan state and artifact index
//!   extraction (no LLM participation)
//!
//! The rotation policy (deciding WHEN to rotate) lives in
//! `agent-works::rotation_policy`. This module handles HOW to store and
//! retrieve state.

pub mod handoff;
pub mod history;
pub mod history_tools;
pub mod notes;
pub mod notes_tools;
pub mod scope;

pub use handoff::{extract_artifact_index, extract_ledger, extract_plan_state};
pub use history::{HistoryStore, read_thread_hint};
pub use history_tools::*;
pub use notes::NotesStore;
pub use notes_tools::*;
pub use scope::SessionScope;
