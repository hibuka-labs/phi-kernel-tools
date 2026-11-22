//! Cross-platform path display helpers.
//!
//! Windows path separators (`\`) leak into every tool result the model
//! reads: a `list_files` line reads `src\main.rs` on Windows and
//! `src/main.rs` on Unix, and a notes listing reads `notes\a.md`. A model
//! that learned the separator from one platform then reads the other
//! wrong — and the agent-facing contract should be one shape everywhere.
//!
//! The helpers here render a path the same way on every platform: `/` as the
//! separator. The *filesystem* path is untouched (Windows APIs still take the
//! native form); only the string the model sees is normalized.

/// Render a path with `/` separators, for display only.
///
/// The path is never used to touch the filesystem in this form — it is a
/// human/model-facing rendering. Callers that need a real path keep the
/// [`std::path::Path`] around.
pub fn display_path(path: &std::path::Path) -> String {
    let raw = path.to_string_lossy();
    if std::path::MAIN_SEPARATOR == '/' {
        return raw.into_owned();
    }
    raw.replace(std::path::MAIN_SEPARATOR, "/")
}

/// Render a path relative to `root`, with `/` separators.
///
/// Falls back to the absolute display path when the path is not under
/// `root` (the caller stripped a prefix it did not own).
pub fn display_relative(root: &std::path::Path, path: &std::path::Path) -> String {
    match path.strip_prefix(root) {
        Ok(rel) => display_path(rel),
        Err(_) => display_path(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn display_path_keeps_unix_separators() {
        assert_eq!(display_path(&PathBuf::from("src/main.rs")), "src/main.rs");
    }

    #[test]
    fn display_relative_falls_back_to_absolute() {
        let root = PathBuf::from("/tmp/root");
        let other = PathBuf::from("/tmp/other/file.txt");
        assert_eq!(display_relative(&root, &other), "/tmp/other/file.txt");
    }

    #[test]
    fn display_relative_strips_the_root() {
        let root = PathBuf::from("/tmp/root");
        let inside = PathBuf::from("/tmp/root/src/main.rs");
        assert_eq!(display_relative(&root, &inside), "src/main.rs");
    }
}
