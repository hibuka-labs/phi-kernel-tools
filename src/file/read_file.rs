use std::path::PathBuf;

use agent_base::{AgentResult, Tool, ToolContext, ToolControlFlow, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::resolve_path;

/// Maximum number of lines returned by default.
const DEFAULT_LIMIT: usize = 2000;

/// Reads a file from the workspace, with optional offset and limit for pagination.
///
/// Paths are resolved relative to the workspace root. Path traversal (`..`) is
/// detected and rejected.
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

    fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace. Returns the file content with line numbers. Supports pagination with offset and limit parameters. Use this to read source code, configuration files, documentation, or any text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file, relative to the workspace root. E.g. 'src/main.rs', 'docs/README.md'."
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
                }
            }
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "Read a file from the workspace with line numbers and pagination support."
                .to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<ToolOutput> {
        let path_str = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();

        if path_str.is_empty() {
            return Ok(ToolOutput {
                summary: "[Error]: No file path provided.".to_string(),
                raw: Some(json!({"error": "no path provided"})),
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
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
                return Ok(ToolOutput {
                    summary: format!("[Error]: {}", e),
                    raw: Some(json!({"error": e, "path": path_str})),
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                });
            }
        };

        // Check if path exists and is a file
        if !file_path.exists() {
            return Ok(ToolOutput {
                summary: format!("[Error]: File not found: {}", path_str),
                raw: Some(json!({"error": "file not found", "path": path_str})),
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
        }

        if !file_path.is_file() {
            return Ok(ToolOutput {
                summary: format!("[Error]: Path is not a file: {}", path_str),
                raw: Some(json!({"error": "not a file", "path": path_str})),
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
        }

        // Read the file
        let content = match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput {
                    summary: format!("[Error]: Failed to read file: {}", e),
                    raw: Some(json!({"error": e.to_string(), "path": path_str})),
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                });
            }
        };

        let all_lines: Vec<&str> = content.lines().collect();
        let total_lines = all_lines.len();

        // Apply offset and limit
        let start = offset.min(total_lines);
        let end = (start + limit).min(total_lines);
        let selected = &all_lines[start..end];

        // Format output with line numbers
        let mut output = String::new();
        output.push_str(&format!(
            "File: {} (lines {}-{} of {})\n",
            path_str,
            if total_lines == 0 { 0 } else { start + 1 },
            end,
            total_lines
        ));

        if selected.is_empty() {
            output.push_str("(file is empty)");
        } else {
            for (i, line) in selected.iter().enumerate() {
                let line_num = start + i + 1;
                output.push_str(&format!("{:>6}|{}\n", line_num, line));
            }
        }

        // Remove trailing newline
        if output.ends_with('\n') {
            output.pop();
        }

        tracing::info!(
            path = %path_str,
            offset = start,
            limit = end - start,
            total = total_lines,
            "read_file"
        );

        Ok(ToolOutput {
            summary: output,
            raw: Some(json!({
                "path": path_str,
                "total_lines": total_lines,
                "start_line": if total_lines == 0 { 0 } else { start + 1 },
                "end_line": end,
                "offset": offset,
                "limit": limit,
                "content": selected.join("\n"),
            })),
            control_flow: ToolControlFlow::Break,
            truncation: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ctx() -> ToolContext {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ToolContext {
            session_id: agent_base::SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: agent_base::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
        }
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

        assert!(result.summary.contains("line1"));
        assert!(result.summary.contains("line2"));
        assert!(result.summary.contains("line3"));
        assert!(result.summary.contains("lines 1-3 of 3"));
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
        assert!(result.summary.contains("line4"));
        assert!(result.summary.contains("line5"));
        assert!(!result.summary.contains("line3"));
        assert!(!result.summary.contains("line6"));
        assert!(result.summary.contains("lines 4-5 of 10"));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"path": "nonexistent.txt"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(result.summary.contains("not found"));
    }

    #[tokio::test]
    async fn test_read_file_is_directory() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let result = tool
            .call(&json!({"path": "subdir"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(result.summary.contains("not a file"));
    }

    #[tokio::test]
    async fn test_read_file_empty() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("empty.txt"), "").unwrap();

        let result = tool
            .call(&json!({"path": "empty.txt"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(result.summary.contains("(file is empty)"));
    }

    #[tokio::test]
    async fn test_read_file_no_path() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool.call(&json!({}), &dummy_ctx()).await.unwrap();

        assert!(result.summary.contains("No file path provided"));
    }

    #[tokio::test]
    async fn test_path_traversal_rejected() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"path": "../etc/passwd"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(result.summary.contains("Error"));
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
        assert!(result.summary.contains("File: short.txt"));
        assert!(!result.summary.contains("only one line"));
    }

    #[tokio::test]
    async fn test_name_and_definition() {
        let tool = ReadFileTool::new(PathBuf::from("/tmp"));
        assert_eq!(tool.name(), "read_file");

        let def = tool.definition();
        assert_eq!(def["function"]["name"], "read_file");
        let params = &def["function"]["parameters"];
        assert_eq!(params["type"], "object");
        let required = params["required"].as_array().unwrap();
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
}
