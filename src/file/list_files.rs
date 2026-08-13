use std::path::PathBuf;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{glob_match, resolve_path};

/// Maximum recursion depth for directory listing.
const MAX_DEPTH: u32 = 64;

/// Lists files and directories in the workspace, with optional glob pattern
/// filtering and recursive mode.
///
/// Paths are resolved relative to the workspace root. Path traversal (`..`) is
/// detected and rejected.
pub struct ListFilesTool {
    workspace_root: PathBuf,
}

impl ListFilesTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }
}

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &'static str {
        "list_files"
    }

    fn description(&self) -> &'static str {
        "List files and directories in a workspace directory. Supports glob pattern filtering (e.g. '*.rs', 'src/**/*.md') and optional recursive mode. Use this to explore project structure, find files by pattern, or understand directory layout."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path, relative to the workspace root. Default: '.' (workspace root)."
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files by name. Supports * (any chars except /) and ? (single char except /). E.g. '*.rs', 'test_*.rs', 'chapter?.md'. Does NOT support ** (use recursive=true for deep listing)."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Set to true to list files recursively in subdirectories. Default: false (single level only)."
                }
            }
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "List files and directories in the workspace with glob filtering and recursive support."
                .to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path_str = args
            .get("path")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| ".".to_string());

        let pattern = args
            .get("pattern")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let recursive = args
            .get("recursive")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        // Resolve and validate the path
        let dir_path = match resolve_path(&self.workspace_root, &path_str) {
            Ok(p) => p,
            Err(e) => {
                return Ok(vec![Content::text(format!("[Error]: {}", e))]);
            }
        };

        // Check if it exists and is a directory
        if !dir_path.exists() {
            return Ok(vec![Content::text(format!(
                "[Error]: Directory not found: {}",
                path_str
            ))]);
        }

        if !dir_path.is_dir() {
            return Ok(vec![Content::text(format!(
                "[Error]: Path is not a directory: {}. Use read_file to read files.",
                path_str
            ))]);
        }

        // Collect entries
        let mut entries: Vec<FileEntry> = Vec::new();

        if recursive {
            collect_entries_recursive(&dir_path, &dir_path, &mut entries, pattern.as_deref(), 0)?;
        } else {
            collect_entries(&dir_path, &dir_path, &mut entries, pattern.as_deref())?;
        }

        // Sort: directories first, then files, alphabetically within each group
        entries.sort_by(|a, b| {
            a.is_dir
                .cmp(&b.is_dir)
                .reverse() // dirs first (true > false)
                .then_with(|| a.name.cmp(&b.name))
        });

        // Format output
        if entries.is_empty() {
            let msg = if pattern.is_some() {
                format!(
                    "Directory '{}' is empty or no entries match the pattern.",
                    path_str
                )
            } else {
                format!("Directory '{}' is empty.", path_str)
            };

            return Ok(vec![Content::text(msg)]);
        }

        let dir_count = entries.iter().filter(|e| e.is_dir).count();
        let file_count = entries.len() - dir_count;

        let mut summary = format!(
            "Listing '{}' ({} files, {} dirs):\n",
            path_str, file_count, dir_count
        );

        for entry in &entries {
            let type_marker = if entry.is_dir { "/" } else { "" };
            let size_str = if entry.is_dir {
                String::new()
            } else {
                format!(" ({})", human_size(entry.size))
            };
            let relative = entry
                .path
                .strip_prefix(&dir_path)
                .unwrap_or(&entry.path)
                .display();
            summary.push_str(&format!("  {}{}{}\n", relative, type_marker, size_str));
        }

        // Remove trailing newline
        if summary.ends_with('\n') {
            summary.pop();
        }

        tracing::info!(
            path = %path_str,
            files = file_count,
            dirs = dir_count,
            recursive = recursive,
            pattern = ?pattern,
            "list_files"
        );

        Ok(vec![Content::text(summary)])
    }
}

struct FileEntry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    size: u64,
}

fn collect_entries(
    base: &std::path::Path,
    dir: &std::path::Path,
    entries: &mut Vec<FileEntry>,
    pattern: Option<&str>,
) -> AgentResult<()> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(d) => d,
        Err(e) => {
            return Err(agent_base::AgentError::internal(format!(
                "Failed to read directory '{}': {}",
                dir.display(),
                e
            )));
        }
    };

    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let entry_path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();

        // Apply pattern filter (only to file name, not full path)
        if let Some(pat) = pattern
            && !glob_match(pat, &file_name)
        {
            continue;
        }

        let is_dir = entry_path.is_dir();
        let size = if is_dir {
            0
        } else {
            entry.metadata().map(|m| m.len()).unwrap_or(0)
        };

        let relative = entry_path.strip_prefix(base).unwrap_or(&entry_path);
        let display_name = relative.display().to_string();

        entries.push(FileEntry {
            name: display_name,
            path: entry_path,
            is_dir,
            size,
        });
    }

    Ok(())
}

fn collect_entries_recursive(
    base: &std::path::Path,
    dir: &std::path::Path,
    entries: &mut Vec<FileEntry>,
    pattern: Option<&str>,
    depth: u32,
) -> AgentResult<()> {
    if depth > MAX_DEPTH {
        tracing::warn!(
            dir = %dir.display(),
            depth = depth,
            max_depth = MAX_DEPTH,
            "list_files: max recursion depth exceeded, stopping"
        );
        return Ok(());
    }

    let read_dir = match std::fs::read_dir(dir) {
        Ok(d) => d,
        Err(e) => {
            // Skip directories we can't read
            tracing::warn!(dir = %dir.display(), error = %e, "list_files: cannot read directory, skipping");
            return Ok(());
        }
    };

    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let entry_path = entry.path();
        let file_type = entry.file_type().ok();
        let is_dir = file_type.as_ref().map(|ft| ft.is_dir()).unwrap_or(false);
        let is_symlink = file_type
            .as_ref()
            .map(|ft| ft.is_symlink())
            .unwrap_or(false);

        if is_dir {
            // Skip symlinked directories to prevent infinite loops
            if is_symlink {
                tracing::debug!(dir = %entry_path.display(), "list_files: skipping symlink directory");
                continue;
            }
            // Recurse into subdirectories
            collect_entries_recursive(base, &entry_path, entries, pattern, depth + 1)?;
        } else {
            let file_name = entry.file_name().to_string_lossy().to_string();

            // Apply pattern filter
            if let Some(pat) = pattern
                && !glob_match(pat, &file_name)
            {
                continue;
            }

            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let relative = entry_path.strip_prefix(base).unwrap_or(&entry_path);
            let display_name = relative.display().to_string();

            entries.push(FileEntry {
                name: display_name,
                path: entry_path,
                is_dir: false,
                size,
            });
        }
    }

    Ok(())
}

/// Format a byte count as a human-readable string.
fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} B", bytes)
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;

    fn dummy_ctx() -> ToolContext {
        ToolContext::for_test()
    }

    fn setup_temp_workspace() -> (tempfile::TempDir, ListFilesTool) {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListFilesTool::new(dir.path().to_path_buf());
        (dir, tool)
    }

    #[tokio::test]
    async fn test_list_files_empty() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"path": "."}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("empty"));
    }

    #[tokio::test]
    async fn test_list_files_with_entries() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.rs"), "b").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let result = tool
            .call(&json!({"path": "."}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("a.txt"));
        assert!(content_text(&result).contains("b.rs"));
        assert!(content_text(&result).contains("sub/"));
        assert!(content_text(&result).contains("2 files, 1 dirs"));
    }

    #[tokio::test]
    async fn test_list_files_with_pattern() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("main.rs"), "main").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "lib").unwrap();
        std::fs::write(dir.path().join("README.md"), "readme").unwrap();

        let result = tool
            .call(&json!({"path": ".", "pattern": "*.rs"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("main.rs"));
        assert!(content_text(&result).contains("lib.rs"));
        assert!(!content_text(&result).contains("README.md"));
    }

    #[tokio::test]
    async fn test_list_files_recursive() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("tests")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "main").unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "lib").unwrap();
        std::fs::write(dir.path().join("tests/test.rs"), "test").unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "toml").unwrap();

        let result = tool
            .call(&json!({"path": ".", "recursive": true}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("src/main.rs"));
        assert!(content_text(&result).contains("src/lib.rs"));
        assert!(content_text(&result).contains("tests/test.rs"));
        assert!(content_text(&result).contains("Cargo.toml"));
    }

    #[tokio::test]
    async fn test_list_files_recursive_with_pattern() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("tests")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "main").unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "lib").unwrap();
        std::fs::write(dir.path().join("src/util.ts"), "ts").unwrap();
        std::fs::write(dir.path().join("tests/test.rs"), "test").unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "toml").unwrap();

        let result = tool
            .call(
                &json!({"path": ".", "recursive": true, "pattern": "*.rs"}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        // Only .rs files should appear
        assert!(content_text(&result).contains("src/main.rs"));
        assert!(content_text(&result).contains("src/lib.rs"));
        assert!(content_text(&result).contains("tests/test.rs"));
        // Non-.rs files should be filtered out
        assert!(!content_text(&result).contains("util.ts"));
        assert!(!content_text(&result).contains("Cargo.toml"));
    }

    #[tokio::test]
    async fn test_list_files_directory_not_found() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"path": "nope"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("not found"));
    }

    #[tokio::test]
    async fn test_list_files_path_is_file() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("file.txt"), "content").unwrap();

        let result = tool
            .call(&json!({"path": "file.txt"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("not a directory"));
    }

    #[tokio::test]
    async fn test_list_files_default_path() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("hello.txt"), "hello").unwrap();

        let result = tool.call(&json!({}), &dummy_ctx()).await.unwrap();

        assert!(content_text(&result).contains("hello.txt"));
    }

    #[tokio::test]
    async fn test_name_and_definition() {
        let tool = ListFilesTool::new(PathBuf::from("/tmp"));
        assert_eq!(tool.name(), "list_files");

        assert_eq!(tool.schema()["type"], "object");
    }

    #[tokio::test]
    async fn test_metadata() {
        let tool = ListFilesTool::new(PathBuf::from("/tmp"));
        let meta = tool.metadata();
        assert_eq!(meta.name, "list_files");
        assert_eq!(meta.origin, "phi-kernel-tools");
        assert!(!meta.description.is_empty());
    }

    #[test]
    fn test_human_size() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(500), "500 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(1536), "1.5 KB");
        assert_eq!(human_size(1048576), "1.0 MB");
    }
}
