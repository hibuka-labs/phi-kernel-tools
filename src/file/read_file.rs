use std::path::{Path, PathBuf};

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::resolve_path;

/// Maximum number of lines returned per file by default.
const DEFAULT_LIMIT: usize = 2000;

/// Smallest remaining budget worth starting another batched file for: enough
/// for a header plus a continuation marker. Below this the batch stops early
/// and names the unread remainder instead of emitting degenerate blocks.
const MIN_BLOCK_BUDGET: usize = 128;

/// Reads one file — or several in one call via `paths` — with optional offset
/// and limit for pagination (applied per file).
///
/// Paths may be workspace-relative or absolute, and may escape the workspace via
/// `..` (no sandbox). Safety is the approval layer.
pub struct ReadFileTool {
    workspace_root: PathBuf,
    /// Opt-in path confinement (see `resolve_path`): reject absolute paths and
    /// `..` escapes from `workspace_root`. Off by default.
    confined: bool,
}

impl ReadFileTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            confined: false,
        }
    }

    /// Enable path confinement: paths must stay inside the workspace root.
    pub fn with_confinement(mut self) -> Self {
        self.confined = true;
        self
    }
}

/// Read one file into its numbered block ("File: … (lines X-Y of Z)" header +
/// numbered body), self-limiting to `budget` chars so the engine's hard reject
/// (§6.5) never fires. `Err(msg)` is the caller-visible `[Error]:` line for a
/// file that cannot be read — batch mode appends it inline and continues.
fn read_block(
    workspace_root: &Path,
    confined: bool,
    path_str: &str,
    offset: usize,
    limit: usize,
    budget: Option<usize>,
) -> Result<String, String> {
    // Resolve and validate the path
    let file_path =
        resolve_path(workspace_root, path_str, confined).map_err(|e| format!("[Error]: {}", e))?;

    if !file_path.exists() {
        return Err(format!("[Error]: File not found: {}", path_str));
    }

    if !file_path.is_file() {
        return Err(format!("[Error]: Path is not a file: {}", path_str));
    }

    // Read file content
    let content = std::fs::read_to_string(&file_path)
        .map_err(|e| format!("[Error]: Failed to read file: {}", e))?;

    let all_lines: Vec<&str> = content.lines().collect();
    let total_lines = all_lines.len();

    // Apply offset and limit
    let start = offset.min(total_lines);
    let end = (start + limit).min(total_lines);
    let selected = &all_lines[start..end];

    // Build the numbered body, self-truncating to the budget. We keep whole
    // lines and append a continuation hint naming the next `offset` to resume
    // from.
    let mut body = String::new();
    let mut emitted = 0usize;
    let mut truncated = false;

    if selected.is_empty() {
        body.push_str("(file is empty)");
    } else {
        // Conservative upper bound on the final header length: the real end
        // line is <= `end`, so this over-estimates by at most a few digits.
        let header_reserve = format!(
            "File: {} (lines {}-{} of {})",
            path_str,
            start + 1,
            end,
            total_lines
        )
        .len()
            + 1; // newline between header and body

        let mut used = 0usize;
        for (i, line) in selected.iter().enumerate() {
            let line_num = start + i + 1;
            let line_str = format!("{:>6}|{}\n", line_num, line);
            let line_chars = line_str.chars().count();

            if let Some(max_chars) = budget {
                let marker = format!("...(truncated, use offset={} to continue)", start + i);
                let reserve = header_reserve + marker.chars().count() + 1; // marker + newline
                if used + line_chars + reserve > max_chars {
                    truncated = true;
                    break;
                }
            }

            body.push_str(&line_str);
            used += line_chars;
            emitted += 1;
        }
    }

    if truncated {
        body.push_str(&format!(
            "...(truncated, use offset={} to continue)\n",
            start + emitted
        ));
    }

    let start_line = if total_lines == 0 { 0 } else { start + 1 };
    let end_line = start + emitted;
    let output = format!(
        "File: {} (lines {start_line}-{end_line} of {total_lines})\n{body}",
        path_str,
    );

    tracing::info!(
        path = %path_str,
        offset = start,
        limit = end - start,
        total = total_lines,
        "read_file"
    );

    Ok(output.trim_end_matches('\n').to_string())
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a file — or several files in one call via `paths`. Returns the file content with line numbers. Supports pagination with offset and limit parameters (applied per file). The path may be workspace-relative or absolute (e.g. a file in another project). Use this to read source code, configuration files, documentation, or any text file. To read several files at once, pass `paths: [..]`: one call returns one numbered block per file."
    }

    fn schema(&self) -> Value {
        let mut props = serde_json::Map::new();
        props.insert(
            "path".to_string(),
            json!({
                "type": "string",
                "description": "Path to one file: workspace-relative (e.g. 'src/main.rs') or absolute (e.g. '/abs/path/file.rs'). Provide exactly one of `path` or `paths`."
            }),
        );
        props.insert(
            "paths".to_string(),
            json!({
                "type": "array",
                "items": { "type": "string" },
                "description": "Batch read: paths to several files (workspace-relative or absolute), all read in one call. Provide exactly one of `path` or `paths`."
            }),
        );
        props.insert(
            "offset".to_string(),
            json!({
                "type": "integer",
                "description": "Line number to start reading from (0-based). Default: 0."
            }),
        );
        props.insert(
            "limit".to_string(),
            json!({
                "type": "integer",
                "description": "Maximum number of lines to read per file. Default: 2000. Set higher if you need more context."
            }),
        );

        json!({
            "type": "object",
            "properties": props
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "Read one or more files (workspace-relative or absolute) with line numbers and pagination support.".to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path_str = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();

        let paths: Vec<String> = args
            .get("paths")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        // Exactly one mode per call: the single `path` or the `paths` batch.
        if !path_str.is_empty() && !paths.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: Provide either `path` or `paths`, not both.".to_string(),
            )]);
        }

        if path_str.is_empty() && paths.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: No file path provided.".to_string(),
            )]);
        }

        let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;

        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT);

        // Single-file mode: one block, byte-identical to the pre-batch shape.
        if !path_str.is_empty() {
            let block = read_block(
                &self.workspace_root,
                self.confined,
                &path_str,
                offset,
                limit,
                ctx.max_output_chars,
            )
            .unwrap_or_else(|e| e);
            return Ok(vec![Content::text(block)]);
        }

        // Batch mode: numbered blocks in order, one shared output budget.
        // A file that cannot be read is an inline `[Error]:` line; the rest of
        // the batch still runs. When the budget runs out, the closing marker
        // names the unread remainder.
        let total = paths.len();
        let mut blocks: Vec<String> = Vec::new();
        let mut used = 0usize;
        let mut stopped_at: Option<usize> = None;

        for (i, p) in paths.iter().enumerate() {
            let remaining = ctx
                .max_output_chars
                .map(|m| m.saturating_sub(used).saturating_sub(2)); // join separator
            if remaining.is_some_and(|r| r < MIN_BLOCK_BUDGET) {
                stopped_at = Some(i);
                break;
            }

            let block = read_block(
                &self.workspace_root,
                self.confined,
                p,
                offset,
                limit,
                remaining,
            )
            .unwrap_or_else(|e| e);
            used += block.chars().count() + 2;
            blocks.push(block);
        }

        if let Some(i) = stopped_at {
            blocks.push(format!(
                "...(output budget reached after {} of {} files — narrow the batch or read the rest in another call)",
                i, total
            ));
        }

        let output = blocks.join("\n\n");
        Ok(vec![Content::text(output)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crate_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn ctx() -> ToolContext {
        ToolContext::for_test()
    }

    fn text(result: &[Content]) -> &str {
        match &result[0] {
            Content::Text { text } => text.as_str(),
            _ => panic!("expected Text content"),
        }
    }

    // ── ReadFileTool basic tests ──

    #[tokio::test]
    async fn read_file_returns_content() {
        let tool = ReadFileTool::new(crate_root());
        let result = tool
            .call(&json!({"path": "Cargo.toml"}), &ctx())
            .await
            .unwrap();
        assert!(text(&result).contains("phi-kernel-tools"));
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let tool = ReadFileTool::new(crate_root());
        let result = tool
            .call(&json!({"path": "nonexistent.txt"}), &ctx())
            .await
            .unwrap();
        assert!(text(&result).contains("[Error]"));
    }

    #[tokio::test]
    async fn read_file_empty_path() {
        let tool = ReadFileTool::new(crate_root());
        let result = tool.call(&json!({"path": ""}), &ctx()).await.unwrap();
        assert!(text(&result).contains("[Error]: No file path provided"));
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let tool = ReadFileTool::new(crate_root());
        let result = tool
            .call(
                &json!({"path": "Cargo.toml", "offset": 2, "limit": 3}),
                &ctx(),
            )
            .await
            .unwrap();
        let content = text(&result);
        assert!(content.contains("lines 3-5 of"));
    }

    #[tokio::test]
    async fn read_file_not_a_file() {
        let tool = ReadFileTool::new(crate_root());
        let result = tool.call(&json!({"path": "src"}), &ctx()).await.unwrap();
        assert!(text(&result).contains("[Error]: Path is not a file"));
    }

    #[tokio::test]
    async fn read_file_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("empty.txt");
        std::fs::write(&file, "").unwrap();
        let tool = ReadFileTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"path": "empty.txt"}), &ctx())
            .await
            .unwrap();
        assert!(text(&result).contains("(file is empty)"));
    }

    #[tokio::test]
    async fn read_file_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("large.txt");
        let content = "line\n".repeat(100);
        std::fs::write(&file, content).unwrap();
        let mut ctx = ToolContext::for_test();
        ctx.max_output_chars = Some(100); // Very small limit to trigger truncation
        let tool = ReadFileTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"path": "large.txt"}), &ctx)
            .await
            .unwrap();
        assert!(text(&result).contains("...(truncated"));
    }

    // ── Batch mode (`paths`) ──

    #[tokio::test]
    async fn read_files_batch_reads_many_in_one_call() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha\nbeta\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "gamma\n").unwrap();
        let tool = ReadFileTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"paths": ["a.txt", "b.txt"]}), &ctx())
            .await
            .unwrap();
        let out = text(&result);
        assert!(out.contains("File: a.txt (lines 1-2 of 2)"));
        assert!(out.contains("alpha"));
        assert!(out.contains("File: b.txt (lines 1-1 of 1)"));
        assert!(out.contains("gamma"));
    }

    #[tokio::test]
    async fn read_files_batch_rejects_both_path_and_paths() {
        let tool = ReadFileTool::new(crate_root());
        let result = tool
            .call(
                &json!({"path": "Cargo.toml", "paths": ["Cargo.toml"]}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(text(&result).contains("[Error]: Provide either `path` or `paths`, not both"));
    }

    #[tokio::test]
    async fn read_files_batch_empty_list_is_no_path() {
        let tool = ReadFileTool::new(crate_root());
        let result = tool.call(&json!({"paths": []}), &ctx()).await.unwrap();
        assert!(text(&result).contains("[Error]: No file path provided"));
    }

    #[tokio::test]
    async fn read_files_batch_reports_missing_files_inline_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha\n").unwrap();
        std::fs::write(dir.path().join("c.txt"), "gamma\n").unwrap();
        let tool = ReadFileTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"paths": ["a.txt", "missing.txt", "c.txt"]}), &ctx())
            .await
            .unwrap();
        let out = text(&result);
        assert!(out.contains("[Error]: File not found: missing.txt"));
        assert!(out.contains("alpha"));
        assert!(out.contains("gamma"));
    }

    #[tokio::test]
    async fn read_files_batch_shares_the_output_budget() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.txt", "b.txt", "c.txt"] {
            std::fs::write(dir.path().join(name), "line\n".repeat(50)).unwrap();
        }
        let mut ctx = ToolContext::for_test();
        ctx.max_output_chars = Some(400);
        let tool = ReadFileTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"paths": ["a.txt", "b.txt", "c.txt"]}), &ctx)
            .await
            .unwrap();
        let out = text(&result);
        assert!(out.contains("...(output budget reached after"));
        assert!(!out.contains("File: c.txt"));
    }

    #[tokio::test]
    async fn read_files_batch_applies_limit_per_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a1\na2\na3\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b1\nb2\nb3\n").unwrap();
        let tool = ReadFileTool::new(dir.path().to_path_buf());
        let result = tool
            .call(&json!({"paths": ["a.txt", "b.txt"], "limit": 2}), &ctx())
            .await
            .unwrap();
        let out = text(&result);
        assert!(out.contains("File: a.txt (lines 1-2 of 3)"));
        assert!(out.contains("File: b.txt (lines 1-2 of 3)"));
        assert!(!out.contains("a3"));
        assert!(!out.contains("b3"));
    }

    #[test]
    fn read_file_metadata() {
        let tool = ReadFileTool::new(crate_root());
        assert_eq!(tool.name(), "read_file");
        assert!(!tool.description().is_empty());
        assert!(!tool.metadata().name.is_empty());
    }

    #[test]
    fn read_file_schema() {
        let tool = ReadFileTool::new(crate_root());
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["paths"].is_object());
        assert!(schema["properties"]["offset"].is_object());
        assert!(schema["properties"]["limit"].is_object());
    }
}
