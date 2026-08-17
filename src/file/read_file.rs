use std::path::PathBuf;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::resolve_path;

/// Maximum number of lines returned by default.
const DEFAULT_LIMIT: usize = 2000;

/// Reads a file, with optional offset and limit for pagination.
///
/// Paths may be workspace-relative or absolute, and may escape the workspace via
/// `..` (no sandbox). Safety is the approval layer.
pub struct ReadFileTool {
    workspace_root: PathBuf,
}

impl ReadFileTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a file. Returns the file content with line numbers. Supports pagination with offset and limit parameters. The path may be workspace-relative or absolute (e.g. a file in another project). Use this to read source code, configuration files, documentation, or any text file."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file: workspace-relative (e.g. 'src/main.rs') or absolute (e.g. '/abs/path/file.rs')."
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Default: 0."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Default: 2000. Set higher if you need more context."
                }
            },
            "required": ["path"]
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "Read a file (workspace-relative or absolute) with line numbers and pagination support."
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
            .unwrap_or("")
            .trim()
            .to_string();

        if path_str.is_empty() {
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

        // Resolve and validate the path
        let file_path = match resolve_path(&self.workspace_root, &path_str) {
            Ok(p) => p,
            Err(e) => {
                return Ok(vec![Content::text(format!("[Error]: {}", e))]);
            }
        };

        // Check if path exists and is a file
        if !file_path.exists() {
            return Ok(vec![Content::text(format!(
                "[Error]: File not found: {}",
                path_str
            ))]);
        }

        if !file_path.is_file() {
            return Ok(vec![Content::text(format!(
                "[Error]: Path is not a file: {}",
                path_str
            ))]);
        }

        // Read the file
        let content = match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(vec![Content::text(format!(
                    "[Error]: Failed to read file: {}",
                    e
                ))]);
            }
        };

        let all_lines: Vec<&str> = content.lines().collect();
        let total_lines = all_lines.len();

        // Apply offset and limit
        let start = offset.min(total_lines);
        let end = (start + limit).min(total_lines);
        let selected = &all_lines[start..end];

        // Build the numbered body, self-truncating to the per-call output
        // budget (`ToolContext::max_output_chars`) so the engine's hard reject
        // (§6.5) never fires for a paginated read. We keep whole lines and
        // append a continuation hint naming the next `offset` to resume from.
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

                if let Some(max_chars) = ctx.max_output_chars {
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
        let mut output = format!(
            "File: {} (lines {start_line}-{end_line} of {total_lines})\n{body}",
            path_str,
        );

        // Remove trailing newline
        while output.ends_with('\n') {
            output.pop();
        }

        tracing::info!(
            path = %path_str,
            offset = start,
            limit = end - start,
            total = total_lines,
            "read_file"
        );

        Ok(vec![Content::text(output)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;

    fn dummy_ctx() -> ToolContext {
        ToolContext::for_test()
    }

    fn setup_temp_workspace() -> (tempfile::TempDir, ReadFileTool) {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(dir.path().to_path_buf());
        (dir, tool)
    }

    #[tokio::test]
    async fn test_read_file_success() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("test.txt"), "line1\nline2\nline3\n").unwrap();

        let result = tool
            .call(&json!({"path": "test.txt"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("line1"));
        assert!(content_text(&result).contains("line2"));
        assert!(content_text(&result).contains("line3"));
        assert!(content_text(&result).contains("lines 1-3 of 3"));
    }

    #[tokio::test]
    async fn test_read_file_with_offset_and_limit() {
        let (dir, tool) = setup_temp_workspace();
        let mut content = String::new();
        for i in 1..=10 {
            content.push_str(&format!("line{}\n", i));
        }
        std::fs::write(dir.path().join("nums.txt"), content).unwrap();

        let result = tool
            .call(
                &json!({"path": "nums.txt", "offset": 3, "limit": 2}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        // offset=3 means lines starting from index 3 (0-based), i.e. line4, line5
        assert!(content_text(&result).contains("line4"));
        assert!(content_text(&result).contains("line5"));
        assert!(!content_text(&result).contains("line3"));
        assert!(!content_text(&result).contains("line6"));
        assert!(content_text(&result).contains("lines 4-5 of 10"));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"path": "nonexistent.txt"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("not found"));
    }

    #[tokio::test]
    async fn test_read_file_is_directory() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let result = tool
            .call(&json!({"path": "subdir"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("not a file"));
    }

    #[tokio::test]
    async fn test_read_file_empty() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("empty.txt"), "").unwrap();

        let result = tool
            .call(&json!({"path": "empty.txt"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("(file is empty)"));
    }

    #[tokio::test]
    async fn test_read_file_no_path() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool.call(&json!({}), &dummy_ctx()).await.unwrap();

        assert!(content_text(&result).contains("No file path provided"));
    }

    #[tokio::test]
    async fn test_read_file_absolute_path_outside_workspace() {
        let (_dir, tool) = setup_temp_workspace();
        // No sandbox: an absolute path outside the workspace is readable.
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("outside.txt"), "secret\n").unwrap();

        let result = tool
            .call(
                &json!({"path": outside.path().join("outside.txt").to_str().unwrap()}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(content_text(&result).contains("secret"));
    }

    #[tokio::test]
    async fn test_read_file_offset_beyond_end() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("short.txt"), "only one line\n").unwrap();

        let result = tool
            .call(&json!({"path": "short.txt", "offset": 10}), &dummy_ctx())
            .await
            .unwrap();

        // offset beyond end should return header only, no content lines
        assert!(content_text(&result).contains("File: short.txt"));
        assert!(!content_text(&result).contains("only one line"));
    }

    #[tokio::test]
    async fn test_name_and_definition() {
        let tool = ReadFileTool::new(PathBuf::from("/tmp"));
        assert_eq!(tool.name(), "read_file");

        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("path")));
    }

    #[tokio::test]
    async fn test_metadata() {
        let tool = ReadFileTool::new(PathBuf::from("/tmp"));
        let meta = tool.metadata();
        assert_eq!(meta.name, "read_file");
        assert_eq!(meta.origin, "phi-kernel-tools");
        assert!(!meta.description.is_empty());
    }

    #[tokio::test]
    async fn test_read_file_self_truncates_to_budget() {
        let (dir, tool) = setup_temp_workspace();
        let content = (1..=500)
            .map(|i| format!("line {i:03} with enough padding to exceed the char budget\n"))
            .collect::<String>();
        std::fs::write(dir.path().join("big.txt"), content).unwrap();

        let mut ctx = dummy_ctx();
        ctx.max_output_chars = Some(400);

        let result = tool.call(&json!({"path": "big.txt"}), &ctx).await.unwrap();
        let text = content_text(&result);

        assert!(
            text.chars().count() <= 400,
            "output exceeds budget: {}",
            text.chars().count()
        );
        assert!(
            text.contains("(truncated, use offset="),
            "missing continuation hint:\n{text}"
        );
        assert!(
            text.contains("line 001"),
            "should include the first line:\n{text}"
        );
        assert!(
            !text.contains("line 500"),
            "should not include the last line:\n{text}"
        );
    }

    #[tokio::test]
    async fn test_read_file_truncation_offset_resumes() {
        let (dir, tool) = setup_temp_workspace();
        let content = (1..=200).map(|i| format!("L{i}\n")).collect::<String>();
        std::fs::write(dir.path().join("big.txt"), content).unwrap();

        let mut ctx = dummy_ctx();
        ctx.max_output_chars = Some(200);

        let first = tool.call(&json!({"path": "big.txt"}), &ctx).await.unwrap();
        let first_text = content_text(&first);
        let offset: usize = first_text
            .split("offset=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("marker should carry a numeric offset");

        let second = tool
            .call(&json!({"path": "big.txt", "offset": offset}), &ctx)
            .await
            .unwrap();
        let second_text = content_text(&second);

        // Continuation's first line is 0-based index `offset`, i.e. 1-based line
        // `offset + 1`, whose content is `L{offset+1}`.
        assert!(
            second_text.contains(&format!("L{}", offset + 1)),
            "continuation should resume at L{0}:\n{second_text}",
            offset + 1
        );
    }
}
