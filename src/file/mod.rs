//! File system kernel tools.
//!
//! Tools that give the LLM the ability to read, write, and list files. By
//! default paths may be workspace-relative or absolute, and may escape the
//! workspace via `..` — there is no path sandbox. Safety comes from the
//! approval layer (`auto`/`ask`/`deny`), matching Claude Code's model. These
//! tools are the foundation for Skills and Memory prompt-injection mode.
//!
//! Opt-in confinement: tools built with [`with_confinement`](ReadFileTool::with_confinement)
//! reject absolute paths and `..` escapes from the workspace root. Off by
//! default — consumers that rely on the approval-layer contract are unchanged.

mod edit_file;
mod list_files;
mod read_file;
mod read_image;
mod url_fetch;
mod write_file;

pub use edit_file::EditFileTool;
pub use list_files::ListFilesTool;
pub use read_file::ReadFileTool;
pub use read_image::ReadImageTool;
pub use url_fetch::FetchPolicy;
pub use write_file::WriteFileTool;

#[cfg(feature = "fuzzing")]
pub mod fuzz {
    pub use super::edit_file::fuzz_exports as edit_file;
    pub use super::list_files::fuzz_exports as list_files;
}

use std::path::{Path, PathBuf};

/// Resolve a user-supplied path to an absolute [`PathBuf`].
///
/// With `confined = false` there is no workspace sandbox: absolute paths are
/// used as-is, relative paths (including `..`) join the workspace root, and
/// existing paths are canonicalized so `.`/`..`/symlinks resolve
/// deterministically. Non-existent paths are left joined (the write tools
/// create them). Safety is the *approval layer* (`auto`/`ask`/`deny`), not
/// path boundaries — matching Claude Code's model.
///
/// With `confined = true` the resolved path must stay inside
/// `workspace_root`: absolute paths and lexical `..` escapes are rejected,
/// and canonicalized existing paths must not escape via symlinks either.
pub(crate) fn resolve_path(
    workspace_root: &Path,
    user_path: &str,
    confined: bool,
) -> Result<PathBuf, String> {
    let trimmed = user_path.trim();
    if trimmed.is_empty() {
        return Err("Empty path provided.".to_string());
    }

    // `~/…` means the home directory. Docs and skill files spell paths that
    // way, and a model reading one should not have to learn the user's home
    // absolute before it can open the file (observed: two wasted tool calls
    // re-finding `~/.phimint/skills/…` by hand). `~user` is not expanded —
    // only the `~` the writer meant as "my home".
    let p = expand_home(trimmed)?;

    if confined {
        if p.is_absolute() {
            return Err(format!(
                "Absolute path '{}' is rejected: this agent is confined to {}.",
                trimmed,
                workspace_root.display()
            ));
        }
        // Lexical check first: reject `..` that would climb above the root,
        // even for paths that don't exist yet.
        normalize_within(workspace_root, &p).map_err(|_| {
            format!(
                "Path '{}' escapes the workspace root {} — rejected.",
                trimmed,
                workspace_root.display()
            )
        })?;
    }

    let resolved = if p.is_absolute() {
        p
    } else {
        workspace_root.join(p)
    };

    if resolved.exists() {
        let canonical = resolved
            .canonicalize()
            .map_err(|e| format!("Failed to resolve path '{}': {}", trimmed, e))?;
        if confined {
            // Symlink escape check: the canonical target must be inside the
            // canonical root as well.
            let root = workspace_root.canonicalize().map_err(|e| {
                format!(
                    "Failed to resolve workspace root '{}': {}",
                    workspace_root.display(),
                    e
                )
            })?;
            if !canonical.starts_with(&root) {
                return Err(format!(
                    "Path '{}' resolves outside the workspace root {} — rejected.",
                    trimmed,
                    workspace_root.display()
                ));
            }
        }
        Ok(canonical)
    } else {
        Ok(resolved)
    }
}

/// Expand a leading `~` / `~/…` to the home directory; every other path comes
/// back as-is. `~user` is deliberately not touched — only the `~` that means
/// "my home" is unambiguous.
fn expand_home(path: &str) -> Result<PathBuf, String> {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return Ok(home_dir()?.join(rest));
    }
    Ok(PathBuf::from(path))
}

fn home_dir() -> Result<PathBuf, String> {
    #[cfg(windows)]
    const HOME_KEY: &str = "USERPROFILE";
    #[cfg(not(windows))]
    const HOME_KEY: &str = "HOME";

    std::env::var_os(HOME_KEY)
        .map(PathBuf::from)
        .filter(|home| !home.as_os_str().is_empty())
        .ok_or_else(|| {
            format!("Cannot expand '~': {HOME_KEY} is not set. Pass the absolute path instead.")
        })
}

/// Lexically normalize `user_path` against `workspace_root`, failing when `..`
/// climbs above the root. Returns the joined path for error-free cases.
fn normalize_within(workspace_root: &Path, user_path: &Path) -> Result<PathBuf, ()> {
    let mut out = workspace_root.to_path_buf();
    for component in user_path.components() {
        match component {
            std::path::Component::Normal(part) => out.push(part),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
                // Climbing above the workspace root is rejected: the popped
                // path no longer starts with the root.
                if !out.starts_with(workspace_root) {
                    return Err(());
                }
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => return Err(()),
        }
    }
    if out.starts_with(workspace_root) {
        Ok(out)
    } else {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn resolve_path_open_mode_allows_absolute_and_escape() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Absolute paths are allowed in open mode. Existing paths come back
        // canonicalized (symlink-safe), so compare against the canonical form
        // — on macOS `/etc` itself is a symlink into `/private/etc`.
        let abs = resolve_path(root, "/etc/hosts", false).unwrap();
        let expected = std::path::PathBuf::from("/etc/hosts");
        let expected = expected.canonicalize().unwrap_or(expected);
        assert_eq!(abs, expected);

        // `..` escapes are allowed in open mode (approval layer is the guard).
        let escaped = resolve_path(root, "../outside.md", false).unwrap();
        assert!(escaped.ends_with("outside.md"));
    }

    #[test]
    fn resolve_path_expands_a_leading_home() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // `~/…` and bare `~` resolve to the home directory — skill files and
        // docs spell paths that way and the model must be able to open them
        // as written. An existing path (the home itself) comes back
        // canonicalized; a fresh one is the plain join.
        let home = home_dir().unwrap();
        assert_eq!(
            resolve_path(root, "~/notes/todo.md", false).unwrap(),
            home.join("notes/todo.md")
        );
        assert_eq!(
            resolve_path(root, "~", false).unwrap(),
            home.canonicalize().unwrap_or_else(|_| home.clone())
        );

        // `~user` is not "my home" — leave it for the join (relative).
        let other = resolve_path(root, "~root/x.md", false).unwrap();
        assert!(other.ends_with("~root/x.md"), "{other:?}");
    }

    #[test]
    fn resolve_path_confined_rejects_absolute() {
        let dir = TempDir::new().unwrap();
        let err = resolve_path(dir.path(), "/etc/hosts", true).unwrap_err();
        assert!(err.contains("rejected"), "unexpected error: {err}");
    }

    #[test]
    fn resolve_path_confined_rejects_dotdot_escape() {
        let dir = TempDir::new().unwrap();
        assert!(resolve_path(dir.path(), "../secret.md", true).is_err());
        assert!(resolve_path(dir.path(), "a/../../secret.md", true).is_err());
    }

    #[test]
    fn resolve_path_confined_allows_inner_dotdot() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("articles");
        std::fs::create_dir_all(root.join("pending")).unwrap();
        let ok = resolve_path(&root, "pending/../pending/draft.md", true).unwrap();
        assert!(ok.ends_with("pending/draft.md"));
    }

    #[test]
    fn resolve_path_confined_rejects_symlink_escape() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.md"), "x").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
        #[cfg(not(unix))]
        return;

        // Lexically inside the root, but canonicalizes outside via symlink.
        assert!(resolve_path(&root, "link/secret.md", true).is_err());
    }
}
