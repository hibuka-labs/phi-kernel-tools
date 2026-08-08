//! File system kernel tools.
//!
//! Tools that give the LLM the ability to read, write, and list files
//! in a workspace-root-sandboxed manner. These are the foundation for
//! Skills and Memory prompt-injection mode.

mod list_files;
mod read_file;
mod write_file;

pub use list_files::ListFilesTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

use std::path::{Path, PathBuf};

/// Resolve a user-supplied path relative to the workspace root.
///
/// Returns the absolute path if it stays within the workspace root.
/// Returns an error string for path traversal attempts or resolution failures.
fn resolve_path(workspace_root: &Path, user_path: &str) -> Result<PathBuf, String> {
    let trimmed = user_path.trim();
    if trimmed.is_empty() {
        return Err("Empty path provided.".to_string());
    }

    // Check the user-supplied path for traversal by analyzing its own components.
    // We do this BEFORE joining with workspace root so we can detect:
    // - absolute paths (starts with /)
    // - parent-dir traversal (.. that goes above the root)
    resolve_user_components(trimmed)?;

    // Join with workspace root — safe after component check
    let resolved = workspace_root.join(trimmed);

    // If the path already exists, verify it's within workspace root via canonicalization
    if resolved.exists() {
        let canonical = resolved
            .canonicalize()
            .map_err(|e| format!("Failed to resolve path '{}': {}", trimmed, e))?;
        let root_canonical = workspace_root
            .canonicalize()
            .map_err(|e| format!("Failed to resolve workspace root: {}", e))?;
        if !canonical.starts_with(&root_canonical) {
            return Err(format!(
                "Path traversal detected: '{}' resolves outside the workspace root.",
                trimmed
            ));
        }
        return Ok(canonical);
    }

    // For non-existent paths, the component check above is sufficient
    Ok(resolved)
}

/// Check user-provided path components for traversal and absolute paths.
fn resolve_user_components(user_path: &str) -> Result<(), String> {
    let path = Path::new(user_path);
    let mut depth: i32 = 0;

    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(format!(
                        "Path traversal detected: '{}' escapes the workspace root.",
                        user_path
                    ));
                }
            }
            std::path::Component::CurDir => {
                // `.` doesn't change depth
            }
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                return Err(format!(
                    "Absolute paths are not allowed: '{}'. Use a relative path.",
                    user_path
                ));
            }
            std::path::Component::Normal(_) => {
                depth += 1;
            }
        }
    }

    Ok(())
}

/// Simple glob pattern matching for file names.
///
/// Supports:
/// - `*` — matches any sequence of characters except `/`
/// - `?` — matches any single character except `/`
///
/// All other characters match literally (case-sensitive).
fn glob_match(pattern: &str, name: &str) -> bool {
    glob_match_inner(pattern.as_bytes(), name.as_bytes())
}

fn glob_match_inner(pat: &[u8], name: &[u8]) -> bool {
    if pat.is_empty() {
        return name.is_empty();
    }

    match pat[0] {
        b'*' => {
            // Try matching zero or more characters (but not '/')
            for i in 0..=name.len() {
                // Stop at '/' boundary — * doesn't cross directories
                if i < name.len() && name[i] == b'/' {
                    break;
                }
                if glob_match_inner(&pat[1..], &name[i..]) {
                    return true;
                }
            }
            false
        }
        b'?' => {
            // Match exactly one character (not '/')
            if name.is_empty() || name[0] == b'/' {
                return false;
            }
            glob_match_inner(&pat[1..], &name[1..])
        }
        _ => {
            if name.is_empty() || pat[0] != name[0] {
                return false;
            }
            glob_match_inner(&pat[1..], &name[1..])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── resolve_path / resolve_user_components ──

    fn setup_root() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    #[test]
    fn test_resolve_path_simple() {
        let (_dir, root) = setup_root();
        let result = resolve_path(&root, "foo.txt").unwrap();
        assert!(result.ends_with("foo.txt"));
        assert!(result.starts_with(&root));
    }

    #[test]
    fn test_resolve_path_nested() {
        let (_dir, root) = setup_root();
        let result = resolve_path(&root, "src/main.rs").unwrap();
        assert!(result.ends_with("src/main.rs"));
    }

    #[test]
    fn test_resolve_path_dot() {
        let (_dir, root) = setup_root();
        let result = resolve_path(&root, ".").unwrap();
        // "." resolves to the workspace root itself
        assert_eq!(result, root.canonicalize().unwrap());
    }

    #[test]
    fn test_resolve_path_dot_slash() {
        let (_dir, root) = setup_root();
        let result = resolve_path(&root, "./foo.txt").unwrap();
        assert!(result.ends_with("foo.txt"));
    }

    #[test]
    fn test_resolve_path_parent_legal() {
        let (_dir, root) = setup_root();
        // foo/../bar stays within root — legal
        let result = resolve_path(&root, "foo/../bar.txt").unwrap();
        assert!(result.ends_with("bar.txt"));
    }

    #[test]
    fn test_resolve_path_dot_nested() {
        let (_dir, root) = setup_root();
        let result = resolve_path(&root, "foo/./bar.txt").unwrap();
        assert!(result.ends_with("bar.txt"));
    }

    #[test]
    fn test_resolve_path_traversal_rejected() {
        let (_dir, root) = setup_root();
        let err = resolve_path(&root, "../etc/passwd").unwrap_err();
        assert!(err.contains("traversal") || err.contains("escapes"));
    }

    #[test]
    fn test_resolve_path_deep_traversal_rejected() {
        let (_dir, root) = setup_root();
        let err = resolve_path(&root, "foo/../../etc/passwd").unwrap_err();
        assert!(err.contains("traversal") || err.contains("escapes"));
    }

    #[test]
    fn test_resolve_path_absolute_rejected() {
        let (_dir, root) = setup_root();
        let err = resolve_path(&root, "/etc/passwd").unwrap_err();
        assert!(err.contains("Absolute") || err.contains("relative"));
    }

    #[test]
    fn test_resolve_path_empty_rejected() {
        let (_dir, root) = setup_root();
        let err = resolve_path(&root, "").unwrap_err();
        assert!(err.contains("Empty"));
    }

    #[test]
    fn test_resolve_path_whitespace_only_rejected() {
        let (_dir, root) = setup_root();
        let err = resolve_path(&root, "   ").unwrap_err();
        assert!(err.contains("Empty"));
    }

    #[test]
    fn test_resolve_path_to_existing_symlink_within_root() {
        let (dir, root) = setup_root();
        std::fs::write(dir.path().join("real.txt"), "hello").unwrap();

        #[cfg(unix)]
        std::os::unix::fs::symlink(dir.path().join("real.txt"), dir.path().join("link.txt"))
            .unwrap();

        // On platforms without symlink support, this test is skipped implicitly
        if dir.path().join("link.txt").exists() {
            let result = resolve_path(&root, "link.txt").unwrap();
            assert!(result.ends_with("real.txt") || result.ends_with("link.txt"));
        }
    }

    #[test]
    fn test_resolve_path_non_existent_returns_unresolved() {
        let (_dir, root) = setup_root();
        // Non-existent path — returns workspace_root.join(user_path) directly
        let result = resolve_path(&root, "new/dir/file.txt").unwrap();
        assert!(result.ends_with("new/dir/file.txt"));
    }

    // ── glob_match ──

    #[test]
    fn test_glob_match_exact() {
        assert!(glob_match("hello.txt", "hello.txt"));
        assert!(!glob_match("hello.txt", "world.txt"));
    }

    #[test]
    fn test_glob_match_star() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("*.rs", "lib.rs"));
        assert!(!glob_match("*.rs", "main.rs.bak"));
        assert!(!glob_match("*.rs", "README.md"));
    }

    #[test]
    fn test_glob_match_question() {
        assert!(glob_match("file?.txt", "file1.txt"));
        assert!(glob_match("file?.txt", "fileA.txt"));
        assert!(!glob_match("file?.txt", "file10.txt"));
        assert!(!glob_match("file?.txt", "file.txt"));
    }

    #[test]
    fn test_glob_match_star_middle() {
        assert!(glob_match("test_*.rs", "test_foo.rs"));
        assert!(glob_match("test_*.rs", "test_bar.rs"));
        assert!(!glob_match("test_*.rs", "test.rs"));
        assert!(!glob_match("test_*.rs", "foo_test.rs"));
    }

    #[test]
    fn test_glob_match_star_does_not_cross_slash() {
        assert!(!glob_match("*.rs", "src/main.rs"));
        assert!(glob_match("*.rs", "main.rs"));
    }

    #[test]
    fn test_glob_match_empty() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "a"));
        assert!(!glob_match("a", ""));
    }
}
