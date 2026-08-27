use std::path::PathBuf;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use ignore::WalkBuilder;
use serde_json::{Value, json};

use super::resolve_path;

// ── Glob matching (module-local) ─────────────────────────────────────────────

/// Simple glob pattern matching for file names.
///
/// Supports:
/// - `*` — matches any sequence of characters except `/`
/// - `?` — matches any single character except `/`
///
/// All other characters match literally (case-sensitive).
fn glob_match(pattern: &str, name: &str) -> bool {
    let pat = pattern.as_bytes();
    let name = name.as_bytes();
    let m = pat.len();
    let n = name.len();

    // DP table: dp[i][j] = glob_match_inner(pat[i..], name[j..])
    // Using 1D rolling array since each row depends only on the next row.
    let mut dp = vec![false; n + 1];
    // Base case: empty pattern matches empty name only
    dp[n] = true;

    // Process pattern right-to-left (skip last byte — handled by base case)
    for i in (0..m).rev() {
        let mut new_dp = vec![false; n + 1];
        match pat[i] {
            b'*' => {
                // * matches any sequence except '/'.
                // dp[i][j] = dp[i+1][j] (match 0) OR dp[i][j+1] (match 1 more)
                // Process right-to-left so dp[i][j+1] is already computed.
                for j in (0..=n).rev() {
                    let skip_slash = j < n && name[j] == b'/';
                    if skip_slash {
                        // * cannot match past '/'
                        new_dp[j] = dp[j]; // match 0 chars
                    } else {
                        // match 0 chars (dp[i+1][j]) OR match 1+ chars (new_dp[j+1])
                        new_dp[j] = dp[j] || (j < n && new_dp[j + 1]);
                    }
                }
            }
            b'?' => {
                // ? matches any single char except '/'
                for j in 0..n {
                    new_dp[j] = name[j] != b'/' && dp[j + 1];
                }
                // j == n: ? can't match empty → new_dp[n] = false (already)
            }
            c => {
                // literal match
                for j in 0..n {
                    new_dp[j] = name[j] == c && dp[j + 1];
                }
            }
        }
        dp = new_dp;
    }

    dp[0]
}

/// Maximum recursion depth for directory listing.
const MAX_DEPTH: usize = 64;

/// Default cap on the number of entries returned by a single listing.
const DEFAULT_LIMIT: usize = 500;

/// Version-control metadata directories, pruned from recursive listings.
///
/// These are not "code-specific" (every git/hg/svn repo has them) — they are
/// universally noise for a file tool, and descending into `.git` would flood a
/// recursive listing with thousands of object/ref files.
const VCS_DIRS: &[&str] = &[".git", ".hg", ".svn"];

/// Lists files and directories, with optional glob pattern filtering and
/// recursive mode.
///
/// Paths may be workspace-relative or absolute (no sandbox).
pub struct ListFilesTool {
    workspace_root: PathBuf,
    /// Entry names (files or directories) to skip. Empty by default — the
    /// framework is domain-agnostic; consumers that know what counts as noise
    /// (e.g. a coding agent ignoring `target/`, `node_modules/`) inject their
    /// own list via [`ListFilesTool::with_excludes`].
    excludes: Vec<String>,
}

impl ListFilesTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self::with_excludes(workspace_root, Vec::new())
    }

    /// Build a tool that skips entries whose name matches any of `excludes`.
    pub fn with_excludes(workspace_root: PathBuf, excludes: Vec<String>) -> Self {
        Self {
            workspace_root,
            excludes,
        }
    }
}

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &'static str {
        "list_files"
    }

    fn description(&self) -> &'static str {
        "List files and directories in a directory. Supports glob pattern filtering (e.g. '*.rs'), optional recursive mode (which respects .gitignore), and a limit on the number of entries returned. The path may be workspace-relative or absolute. Use this to explore project structure, find files by pattern, or understand directory layout."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path: workspace-relative or absolute. Default: '.' (workspace root)."
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files by name. Supports * (any chars except /) and ? (single char except /). E.g. '*.rs', 'test_*.rs', 'chapter?.md'. Does NOT support ** (use recursive=true for deep listing)."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Set to true to list files recursively in subdirectories. Default: false (single level only). In recursive mode, .gitignore/.ignore rules are respected and version-control metadata dirs (.git/.hg/.svn) are skipped."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to return (default: 500). Raise it to list more, or narrow path/pattern."
                }
            }
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "List files and directories (workspace-relative or absolute) with glob filtering and recursive support."
                .to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
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

        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_LIMIT);

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

        let limit_reached = if recursive {
            collect_entries_recursive(
                &dir_path,
                &mut entries,
                pattern.as_deref(),
                &self.excludes,
                limit,
            )?
        } else {
            collect_entries(
                &dir_path,
                &mut entries,
                pattern.as_deref(),
                &self.excludes,
                limit,
            )?
        };

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

        let header = format!(
            "Listing '{}' ({} files, {} dirs):\n",
            path_str, file_count, dir_count
        );
        let marker = "...(truncated)\n";

        let mut summary = header.clone();
        let mut truncated = false;

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
            let line = format!("  {}{}{}\n", relative, type_marker, size_str);

            // Self-truncate to the per-call output budget (`ToolContext::max_output_chars`)
            // so the engine's hard reject (§6.5) never fires for a huge listing. Unlike
            // read_file there is no natural "offset" to resume from, so the hint just
            // tells the agent to narrow `path`/`pattern` and re-list.
            if let Some(max_chars) = ctx.max_output_chars
                && summary.chars().count() + line.chars().count() + marker.chars().count()
                    > max_chars
            {
                truncated = true;
                break;
            }
            summary.push_str(&line);
        }

        if truncated {
            summary.push_str(marker);
        }

        // Remove trailing newline
        if summary.ends_with('\n') {
            summary.pop();
        }

        // When the entry cap fired, tell the agent how to get more. This is a
        // *scope* hint (files have no meaningful order, so there is no
        // offset-style pagination like read_file) — either raise the cap or
        // narrow `path`/`pattern`.
        if limit_reached {
            summary.push_str(&format!(
                "\n\n[{} entries limit reached. use limit={} for more, or narrow path/pattern]",
                limit,
                limit.saturating_mul(2),
            ));
        }

        tracing::info!(
            path = %path_str,
            files = file_count,
            dirs = dir_count,
            recursive = recursive,
            pattern = ?pattern,
            limit_reached = limit_reached,
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

/// Collect a single directory level (shallow listing). Returns `true` when the
/// `limit` cap was hit (i.e. there may be more entries).
fn collect_entries(
    dir: &std::path::Path,
    entries: &mut Vec<FileEntry>,
    pattern: Option<&str>,
    excludes: &[String],
    limit: usize,
) -> AgentResult<bool> {
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

        // Skip consumer-injected excludes (e.g. build output, VCS dirs).
        if excludes.iter().any(|e| e == &file_name) {
            continue;
        }

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

        let relative = entry_path.strip_prefix(dir).unwrap_or(&entry_path);
        let display_name = relative.display().to_string();

        entries.push(FileEntry {
            name: display_name,
            path: entry_path,
            is_dir,
            size,
        });

        if entries.len() >= limit {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Recursively collect files (not directories) under `dir`, driven by the
/// [`ignore`](https://crates.io/crates/ignore) walker so that `.gitignore` /
/// `.ignore` rules are respected and version-control metadata dirs are pruned.
///
/// Returns `true` when the `limit` cap was hit (i.e. there may be more entries).
fn collect_entries_recursive(
    dir: &std::path::Path,
    entries: &mut Vec<FileEntry>,
    pattern: Option<&str>,
    excludes: &[String],
    limit: usize,
) -> AgentResult<bool> {
    let filter_excludes: Vec<String> = excludes.to_vec();

    let mut builder = WalkBuilder::new(dir);
    builder
        .hidden(false) // include dotfiles (.env, .gitignore, ...)
        .git_ignore(true) // respect .gitignore
        .git_global(false) // skip the user's global ignore for determinism
        .git_exclude(false) // repo-local .git/info/exclude only; may not be a repo
        .ignore(true) // respect .ignore files
        .parents(true) // apply parent .gitignore (matches git semantics for subdirs)
        .follow_links(false) // never follow symlinks (avoids cycles)
        .require_git(false) // respect .gitignore even outside a git repo
        .max_depth(Some(MAX_DEPTH))
        .filter_entry(move |entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            // Prune VCS metadata and consumer excludes before descending, so a
            // huge `target/` or `.git/` is never walked at all.
            !VCS_DIRS.iter().any(|v| *v == name) && !filter_excludes.iter().any(|e| e == &name)
        });

    for result in builder.build() {
        let entry = match result {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, dir = %dir.display(), "list_files: walk error, skipping");
                continue;
            }
        };

        // Recursive mode lists files only — directories are implied by their
        // path prefixes, matching `find`-style semantics. The walker still
        // descends through directories; we just don't emit them.
        let Some(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }

        let entry_path = entry.path().to_path_buf();
        let file_name = entry
            .path()
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Apply pattern filter (only to file name, not full path)
        if let Some(pat) = pattern
            && !glob_match(pat, &file_name)
        {
            continue;
        }

        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let relative = entry_path.strip_prefix(dir).unwrap_or(&entry_path);
        let display_name = relative.display().to_string();

        entries.push(FileEntry {
            name: display_name,
            path: entry_path,
            is_dir: false,
            size,
        });

        if entries.len() >= limit {
            return Ok(true);
        }
    }

    Ok(false)
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

// ── Fuzz exports ──
#[cfg(feature = "fuzzing")]
pub mod fuzz_exports {
    pub fn glob_match(pattern: &str, name: &str) -> bool {
        super::glob_match(pattern, name)
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

    #[tokio::test]
    async fn test_list_files_excludes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("target")).unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("target/build.o"), "obj").unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "main").unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "toml").unwrap();

        let tool =
            ListFilesTool::with_excludes(dir.path().to_path_buf(), vec!["target".to_string()]);

        let result = tool
            .call(&json!({"path": ".", "recursive": true}), &dummy_ctx())
            .await
            .unwrap();
        let text = content_text(&result);

        assert!(text.contains("src/main.rs"), "{text}");
        assert!(text.contains("Cargo.toml"), "{text}");
        assert!(!text.contains("target"), "{text}");
    }

    #[tokio::test]
    async fn test_list_files_self_truncates_to_budget() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..200 {
            std::fs::write(
                dir.path().join(format!("file_{i:03}_with_a_long_name.rs")),
                "x",
            )
            .unwrap();
        }

        let mut ctx = dummy_ctx();
        ctx.max_output_chars = Some(200);
        let tool = ListFilesTool::new(dir.path().to_path_buf());

        let result = tool
            .call(&json!({"path": ".", "recursive": true}), &ctx)
            .await
            .unwrap();
        let text = content_text(&result);

        assert!(
            text.chars().count() <= 200,
            "output exceeds budget: {}",
            text.chars().count()
        );
        assert!(
            text.contains("...(truncated)"),
            "missing truncation hint:\n{text}"
        );
    }

    #[test]
    fn test_human_size() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(500), "500 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(1536), "1.5 KB");
        assert_eq!(human_size(1048576), "1.0 MB");
    }

    #[tokio::test]
    async fn test_list_files_recursive_respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("skipme")).unwrap();
        std::fs::write(dir.path().join("skipme/generated.o"), "obj").unwrap();
        std::fs::write(dir.path().join("keep.rs"), "keep").unwrap();
        // A .gitignore in a non-git dir still applies (require_git(false)).
        std::fs::write(dir.path().join(".gitignore"), "skipme/\n").unwrap();

        let tool = ListFilesTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"path": ".", "recursive": true}), &dummy_ctx())
            .await
            .unwrap();
        let text = content_text(&result);

        assert!(text.contains("keep.rs"), "{text}");
        assert!(
            !text.contains("skipme"),
            "gitignored dir leaked through:\n{text}"
        );
    }

    #[tokio::test]
    async fn test_list_files_limit() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..20 {
            std::fs::write(dir.path().join(format!("f{i:02}.txt")), "x").unwrap();
        }

        let tool = ListFilesTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"path": ".", "limit": 5}), &dummy_ctx())
            .await
            .unwrap();
        let text = content_text(&result);

        // The cap fired, the header reports the capped count, and the notice
        // names both the current and suggested next limit.
        assert!(text.contains("5 entries limit reached"), "{text}");
        assert!(text.contains("limit=10"), "{text}");
        assert!(text.contains("(5 files, 0 dirs)"), "{text}");
    }

    #[test]
    fn glob_match_backtracking_does_not_hang() {
        // Pattern with alternating *? creates exponential backtracking in naive
        // recursive implementation when NO match exists (must explore all branches).
        // Use 'b' suffix so it never matches 'a' input → forces full search.
        let pattern = "*?*?*?*?*?*?*?*?*?*?b";
        let name = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; // 40 a's

        let start = std::time::Instant::now();
        let result = glob_match(pattern, name);
        let elapsed = start.elapsed();

        eprintln!("glob_match({:?}, {:?}) took {:?}", pattern, name, elapsed);

        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "glob_match took {:?} — exponential backtracking detected",
            elapsed,
        );
        assert!(!result, "pattern should not match");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn glob_match_star_matches_everything(name in "[a-z]{0,50}") {
            // "*" should match any name (without '/')
            assert!(glob_match("*", &name));
        }

        #[test]
        fn glob_match_question_mark_matches_single_char(c in b'a'..=b'z') {
            let name = String::from_utf8(vec![c]).unwrap();
            assert!(glob_match("?", &name));
        }

        #[test]
        fn glob_match_does_not_panic(pattern in "[*?a-z]{0,30}", name in "[a-z]{0,50}") {
            let _ = glob_match(&pattern, &name);
        }

        #[test]
        fn glob_match_star_prefix_matches_any_suffix(
            prefix in "[a-z]{1,5}",
            suffix in "[a-z]{0,20}"
        ) {
            let pattern = format!("{}*", prefix);
            let name = format!("{}{}", prefix, suffix);
            assert!(glob_match(&pattern, &name));
        }

        #[test]
        fn human_size_does_not_panic(bytes in 0..u64::MAX) {
            let result = human_size(bytes);
            assert!(!result.is_empty());
        }
    }
}
