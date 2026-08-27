//! File system kernel tools.
//!
//! Tools that give the LLM the ability to read, write, and list files. Paths
//! may be workspace-relative or absolute, and may escape the workspace via
//! `..` — there is no path sandbox. Safety comes from the approval layer
//! (`auto`/`ask`/`deny`), matching Claude Code's model. These tools are the
//! foundation for Skills and Memory prompt-injection mode.

mod edit_file;
mod list_files;
mod read_file;
mod write_file;

pub use edit_file::EditFileTool;
pub use list_files::ListFilesTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

#[cfg(feature = "fuzzing")]
pub mod fuzz {
    pub use super::edit_file::fuzz_exports as edit_file;
    pub use super::list_files::fuzz_exports as list_files;
}

use std::path::{Path, PathBuf};

/// Resolve a user-supplied path to an absolute [`PathBuf`].
///
/// There is no workspace sandbox: absolute paths are used as-is, relative paths
/// (including `..`) join the workspace root, and existing paths are canonicalized
/// so `.`/`..`/symlinks resolve deterministically. Non-existent paths are left
/// joined (the write tools create them). Safety is the *approval layer*
/// (`auto`/`ask`/`deny`), not path boundaries — matching Claude Code's model.
pub(crate) fn resolve_path(workspace_root: &Path, user_path: &str) -> Result<PathBuf, String> {
    let trimmed = user_path.trim();
    if trimmed.is_empty() {
        return Err("Empty path provided.".to_string());
    }

    let p = Path::new(trimmed);
    let resolved = if p.is_absolute() {
        p.to_path_buf()
    } else {
        workspace_root.join(p)
    };

    if resolved.exists() {
        resolved
            .canonicalize()
            .map_err(|e| format!("Failed to resolve path '{}': {}", trimmed, e))
    } else {
        Ok(resolved)
    }
}
