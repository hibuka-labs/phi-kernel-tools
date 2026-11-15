//! Context rotation — tools and I/O for managing context window lifecycle.
//!
//! This module provides the tools that the LLM uses to persist state across
//! context window rotations. Each store gets the verbs that need it and no
//! more — see `notes_tools::create_notes_tools` and `history_tool`:
//!
//! - **notes**: the per-session scratchpad (`notes_read_file`, `notes_write_file`)
//!   — two tools, because only reading and writing need the store to resolve a
//!   path the prompt cannot name. Listing folds into `notes_read_file`; there is
//!   no search, because a scratchpad is found by name.
//! - **history**: the archived windows, one tool (`history`, `op` selects
//!   list_windows / list_items / read / search) — structured data, so not plain
//!   files, but one question asked four ways does not need four names.
//! - **handoff**: mechanical activity ledger, plan state and artifact index
//!   extraction (no LLM participation)

pub mod handoff;
pub mod history;
pub mod history_tool;
pub mod history_tools;
pub mod notes;
pub mod notes_tools;
pub mod scope;

pub use handoff::{extract_artifact_index, extract_ledger, extract_plan_state};
pub use history::{HistoryStore, read_thread_hint};
pub use history_tool::HistoryTool;
pub use history_tools::*;
pub use notes::NotesStore;
pub use notes_tools::*;
pub use scope::SessionScope;

#[cfg(test)]
mod description_guard {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;

    /// Value: protects=the tool-choice guidance the rotation tools carry — the
    /// notes and history stores exist because the paths and the structure are
    /// not knowable from a prompt, not because a verb deserved a tool name;
    /// a description that forgets to say when NOT to use one puts the model
    /// back to guessing. fails_when=a description is cut to a one-liner or
    /// loses its routing markers. why_new=both tool sets were collapsed and
    /// their descriptions rewritten. seam=description text; no_claim=that the
    /// two-set split is optimal, only that each tool must say when to stop.
    #[test]
    fn rotation_tool_descriptions_route_and_do_not_shrink() {
        let tmp = TempDir::new().unwrap();
        let notes = create_notes_tools(Arc::new(NotesStore::new(tmp.path(), "test", "main")));
        let history = create_history_tools(Arc::new(HistoryStore::new(tmp.path(), "test")));
        for tool in notes.into_iter().chain(history) {
            let d = tool.description();
            assert!(
                d.chars().count() >= 400,
                "{}: description shrank to a one-liner ({} chars)",
                tool.name(),
                d.chars().count()
            );
            assert!(
                d.contains("Do NOT") || d.contains("Do not") || d.contains("only when"),
                "{}: must say when NOT to use it: {d}",
                tool.name()
            );
        }
    }
}
