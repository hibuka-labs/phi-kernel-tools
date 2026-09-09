//! Context rotation — tools and I/O for managing context window lifecycle.
//!
//! This module provides the tools that the LLM uses to persist state across
//! context window rotations:
//!
//! - **notes**: persistent scratchpad (`notes.read_file`, `notes.write_file`,
//!   `notes.list_files`, `notes.search_contents`, `notes.append_to_file`)
//! - **history**: archived window access (`history.list_windows`,
//!   `history.list_items`, `history.read_item`, `history.search_contents`)
//! - **handoff**: mechanical activity ledger extraction (no LLM participation)
//!
//! The rotation policy (deciding WHEN to rotate) lives in
//! `agent-works::rotation_policy`. This module handles HOW to store and
//! retrieve state.

pub mod handoff;
pub mod history;
pub mod history_tools;
pub mod notes;
pub mod notes_tools;

pub use handoff::extract_ledger;
pub use history::{HistoryStore, read_thread_hint};
pub use history_tools::*;
pub use notes::NotesStore;
pub use notes_tools::*;
