use std::path::PathBuf;

use agent_base::{AgentResult, Tool, ToolContext, ToolControlFlow, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::resolve_path;

/// Maximum file size for writes (1 MB).
const MAX_FILE_SIZE: usize = 1_048_576;

/// Writes or creates a file in the workspace.
///
/// Paths are resolved relative to the workspace root. Path traversal (`..`) is
/// detected and rejected. By default, existing files are not overwritten unless
/// `overwrite: true` is set.
pub struct WriteFileTool {
    workspace_root: PathBuf,
}

impl WriteFileTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write or create a file in the workspace. Creates parent directories automatically. Will not overwrite existing files unless 'overwrite' is set to true. Content size is limited to 1 MB. Use this to create or update source files, configuration, documentation, or any text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file, relative to the workspace root. E.g. 'src/main.rs', 'config.toml'. Parent directories will be created if needed."
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file."
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Set to true to overwrite an existing file. Default: false."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "Write or create a file in the workspace, with path sandboxing and overwrite protection."
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

        let content = args
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let overwrite = args
            .get("overwrite")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        if path_str.is_empty() {
            return Ok(ToolOutput {
                summary: "[Error]: No file path provided.".to_string(),
                raw: Some(json!({"error": "no path provided"})),
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
        }

        // Check content size limit
        if content.len() > MAX_FILE_SIZE {
            return Ok(ToolOutput {
                summary: format!(
                    "[Error]: Content size ({} bytes) exceeds the maximum allowed size ({} bytes / ~1 MB).",
                    content.len(),
                    MAX_FILE_SIZE
                ),
                raw: Some(json!({
                    "error": "content too large",
                    "content_size": content.len(),
                    "max_size": MAX_FILE_SIZE,
                    "path": path_str,
                })),
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
        }

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

        // Check if file already exists
        if file_path.exists() {
            if file_path.is_dir() {
                return Ok(ToolOutput {
                    summary: format!("[Error]: Path is a directory, not a file: {}", path_str),
                    raw: Some(json!({"error": "path is a directory", "path": path_str})),
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                });
            }
            if !overwrite {
                return Ok(ToolOutput {
                    summary: format!(
                        "[Error]: File already exists: {}. Use overwrite=true to replace it.",
                        path_str
                    ),
                    raw: Some(json!({
                        "error": "file already exists",
                        "path": path_str,
                        "hint": "set overwrite=true to replace",
                    })),
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                });
            }
        }

        // Create parent directories
        if let Some(parent) = file_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Ok(ToolOutput {
                    summary: format!("[Error]: Failed to create parent directories: {}", e),
                    raw: Some(json!({
                        "error": e.to_string(),
                        "path": path_str,
                    })),
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                });
            }
        }

        // Write the file
        match std::fs::write(&file_path, &content) {
            Ok(()) => {
                let line_count = content.lines().count();
                let verb = if file_path.exists() && overwrite {
                    "Updated"
                } else {
                    "Created"
                };

                tracing::info!(
                    path = %path_str,
                    size = content.len(),
                    lines = line_count,
                    overwrite = overwrite,
                    "write_file"
                );

                Ok(ToolOutput {
                    summary: format!(
                        "{} file: {} ({} bytes, {} lines)",
                        verb,
                        path_str,
                        content.len(),
                        line_count
                    ),
                    raw: Some(json!({
                        "path": path_str,
                        "size": content.len(),
                        "lines": line_count,
                        "overwrite": overwrite,
                        "created": !overwrite,
                    })),
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                })
            }
            Err(e) => Ok(ToolOutput {
                summary: format!("[Error]: Failed to write file: {}", e),
                raw: Some(json!({
                    "error": e.to_string(),
                    "path": path_str,
                })),
                control_flow: ToolControlFlow::Break,
                truncation: None,
            }),
        }
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

    fn setup_temp_workspace() -> (tempfile::TempDir, WriteFileTool) {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(dir.path().to_path_buf());
        (dir, tool)
    }

    #[tokio::test]
    async fn test_write_file_create() {
        let (dir, tool) = setup_temp_workspace();

        let result = tool
            .call(
                &json!({"path": "new.txt", "content": "hello world"}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(result.summary.contains("Created"));
        assert!(result.summary.contains("new.txt"));

        let content = std::fs::read_to_string(dir.path().join("new.txt")).unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn test_write_file_overwrite_false_by_default() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("existing.txt"), "original").unwrap();

        let result = tool
            .call(
                &json!({"path": "existing.txt", "content": "replaced"}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(result.summary.contains("already exists"));
        assert!(result.summary.contains("overwrite=true"));

        // File should be unchanged
        let content = std::fs::read_to_string(dir.path().join("existing.txt")).unwrap();
        assert_eq!(content, "original");
    }

    #[tokio::test]
    async fn test_write_file_overwrite_true() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::write(dir.path().join("existing.txt"), "original").unwrap();

        let result = tool
            .call(
                &json!({"path": "existing.txt", "content": "replaced", "overwrite": true}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(result.summary.contains("Updated"));
        assert!(result.summary.contains("existing.txt"));

        let content = std::fs::read_to_string(dir.path().join("existing.txt")).unwrap();
        assert_eq!(content, "replaced");
    }

    #[tokio::test]
    async fn test_write_file_creates_parent_dirs() {
        let (dir, tool) = setup_temp_workspace();

        let result = tool
            .call(
                &json!({"path": "nested/deep/file.txt", "content": "deep"}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(result.summary.contains("Created"));
        assert!(dir.path().join("nested/deep/file.txt").exists());
    }

    #[tokio::test]
    async fn test_write_file_content_too_large() {
        let (_dir, tool) = setup_temp_workspace();
        let big_content = "x".repeat(MAX_FILE_SIZE + 1);

        let result = tool
            .call(
                &json!({"path": "big.txt", "content": big_content}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(result.summary.contains("exceeds the maximum"));
    }

    #[tokio::test]
    async fn test_write_file_no_path() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"content": "stuff"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(result.summary.contains("No file path provided"));
    }

    #[tokio::test]
    async fn test_path_traversal_rejected() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(
                &json!({"path": "../outside.txt", "content": "evil"}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(result.summary.contains("Error"));
    }

    #[tokio::test]
    async fn test_write_to_directory_rejected() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::create_dir(dir.path().join("mydir")).unwrap();

        let result = tool
            .call(&json!({"path": "mydir", "content": "stuff"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(result.summary.contains("directory"));
    }

    #[tokio::test]
    async fn test_write_file_empty_content() {
        let (dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"path": "empty.txt", "content": ""}), &dummy_ctx())
            .await
            .unwrap();

        assert!(result.summary.contains("Created"));
        assert!(result.summary.contains("0 bytes"));
        assert!(result.summary.contains("0 lines"));
        assert!(dir.path().join("empty.txt").exists());
    }

    #[tokio::test]
    async fn test_name_and_definition() {
        let tool = WriteFileTool::new(PathBuf::from("/tmp"));
        assert_eq!(tool.name(), "write_file");

        let def = tool.definition();
        assert_eq!(def["function"]["name"], "write_file");
        let required = def["function"]["parameters"]["required"]
            .as_array()
            .unwrap();
        let required_names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_names.contains(&"path"));
        assert!(required_names.contains(&"content"));
    }

    #[tokio::test]
    async fn test_metadata() {
        let tool = WriteFileTool::new(PathBuf::from("/tmp"));
        let meta = tool.metadata();
        assert_eq!(meta.name, "write_file");
        assert_eq!(meta.origin, "phi-kernel-tools");
        assert!(!meta.description.is_empty());
    }
}
