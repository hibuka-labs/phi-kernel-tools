//! The live session a session-keyed store serves.
//!
//! History, notes, code journals and the rotation archive are all keyed by
//! session id. They are built once (at agent build) but must follow a
//! `/resume` session switch: a store frozen on the launch session keeps
//! writing one session's archives into another's directory (observed:
//! 2026-11-04 — a mid-session switch left the rotation archive and the code
//! journal under the abandoned launch session). Every such store holds this
//! scope instead of a copied id and resolves its paths at use time; the
//! product re-points it once per session (`session_boot`).

use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Base data root plus the current session id — shared, re-pointable.
pub struct SessionScope {
    base_dir: PathBuf,
    current: RwLock<String>,
}

impl SessionScope {
    /// A scope serving `session_id` under `base_dir` (typically `~/.phimint/`).
    pub fn new(base_dir: impl Into<PathBuf>, session_id: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
            current: RwLock::new(session_id.into()),
        }
    }

    /// The session this scope currently serves.
    pub fn session_id(&self) -> String {
        self.current
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Serve a different session from now on (the `/resume` switch).
    pub fn repoint(&self, session_id: impl Into<String>) {
        *self.current.write().unwrap_or_else(|e| e.into_inner()) = session_id.into();
    }

    /// The data root the scope was created under (never re-points).
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// `<base_dir>/sessions/<session_id>/` — live session files (messages,
    /// code journal, captures).
    pub fn session_dir(&self) -> PathBuf {
        self.base_dir.join("sessions").join(self.session_id())
    }

    /// `<base_dir>/history/<session_id>/` — archived context windows.
    pub fn history_dir(&self) -> PathBuf {
        self.base_dir.join("history").join(self.session_id())
    }

    /// `<base_dir>/notes/<session_id>/<agent_name>/` — the agent's notes.
    pub fn notes_dir(&self, agent_name: &str) -> PathBuf {
        self.base_dir
            .join("notes")
            .join(self.session_id())
            .join(agent_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repoint_moves_every_tree() {
        let scope = SessionScope::new("/root", "sess_a");
        assert_eq!(scope.session_id(), "sess_a");
        assert_eq!(scope.session_dir(), PathBuf::from("/root/sessions/sess_a"));
        assert_eq!(scope.history_dir(), PathBuf::from("/root/history/sess_a"));
        assert_eq!(
            scope.notes_dir("main"),
            PathBuf::from("/root/notes/sess_a/main")
        );

        scope.repoint("sess_b");
        assert_eq!(scope.session_id(), "sess_b");
        assert_eq!(scope.session_dir(), PathBuf::from("/root/sessions/sess_b"));
        assert_eq!(scope.history_dir(), PathBuf::from("/root/history/sess_b"));
        assert_eq!(
            scope.notes_dir("main"),
            PathBuf::from("/root/notes/sess_b/main")
        );
        // The base root never moves.
        assert_eq!(scope.base_dir(), Path::new("/root"));
    }
}
