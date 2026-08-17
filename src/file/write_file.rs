use std::path::PathBuf;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::resolve_path;

/// Maximum file size for writes (1 MB).
const MAX_FILE_SIZE: usize = 1_048_576;

/// Writes or creates a file.
///
/// Paths may be workspace-relative or absolute (no sandbox). By default,
/// existing files are not overwritten unless `overwrite: true` is set.
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

    fn description(&self) -> &'static str {
        "Write or create a file. Creates parent directories automatically. Will not overwrite existing files unless 'overwrite' is set to true. Content size is limited to 1 MB. The path may be workspace-relative or absolute. Use this to create or update source files, configuration, documentation, or any text file."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file: workspace-relative (e.g. 'src/main.rs') or absolute. Parent directories will be created if needed."
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
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "Write or create a file (workspace-relative or absolute) with overwrite protection."
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
            return Ok(vec![Content::text(
                "[Error]: No file path provided.".to_string(),
            )]);
        }

        // Check content size limit
        if content.len() > MAX_FILE_SIZE {
            return Ok(vec![Content::text(format!(
                "[Error]: Content size ({} bytes) exceeds the maximum allowed size ({} bytes / ~1 MB).",
                content.len(),
                MAX_FILE_SIZE
            ))]);
        }

        // Resolve and validate the path
        let file_path = match resolve_path(&self.workspace_root, &path_str) {
            Ok(p) => p,
            Err(e) => {
                return Ok(vec![Content::text(format!("[Error]: {}", e))]);
            }
        };

        // Check if file already exists
        if file_path.exists() {
            if file_path.is_dir() {
                return Ok(vec![Content::text(format!(
                    "[Error]: Path is a directory, not a file: {}",
                    path_str
                ))]);
            }
            if !overwrite {
                return Ok(vec![Content::text(format!(
                    "[Error]: File already exists: {}. Use overwrite=true to replace it.",
                    path_str
                ))]);
            }
        }

        // Create parent directories
        if let Some(parent) = file_path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return Ok(vec![Content::text(format!(
                "[Error]: Failed to create parent directories: {}",
                e
            ))]);
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

                Ok(vec![Content::text(format!(
                    "{} file: {} ({} bytes, {} lines)",
                    verb,
                    path_str,
                    content.len(),
                    line_count
                ))])
            }
            Err(e) => Ok(vec![Content::text(format!(
                "[Error]: Failed to write file: {}",
                e
            ))]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;

    fn dummy_ctx() -> ToolContext {
        ToolContext::for_test()
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

        assert!(content_text(&result).contains("Created"));
        assert!(content_text(&result).contains("new.txt"));

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

        assert!(content_text(&result).contains("already exists"));
        assert!(content_text(&result).contains("overwrite=true"));

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

        assert!(content_text(&result).contains("Updated"));
        assert!(content_text(&result).contains("existing.txt"));

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

        assert!(content_text(&result).contains("Created"));
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

        assert!(content_text(&result).contains("exceeds the maximum"));
    }

    #[tokio::test]
    async fn test_write_file_no_path() {
        let (_dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"content": "stuff"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("No file path provided"));
    }

    #[tokio::test]
    async fn test_write_file_absolute_path_outside_workspace() {
        let (_dir, tool) = setup_temp_workspace();
        // No sandbox: an absolute path outside the workspace is writable.
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("outside.txt");

        let result = tool
            .call(
                &json!({"path": target.to_str().unwrap(), "content": "external"}),
                &dummy_ctx(),
            )
            .await
            .unwrap();

        assert!(content_text(&result).contains("Created"));
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "external");
    }

    #[tokio::test]
    async fn test_write_to_directory_rejected() {
        let (dir, tool) = setup_temp_workspace();
        std::fs::create_dir(dir.path().join("mydir")).unwrap();

        let result = tool
            .call(&json!({"path": "mydir", "content": "stuff"}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("directory"));
    }

    #[tokio::test]
    async fn test_write_file_empty_content() {
        let (dir, tool) = setup_temp_workspace();

        let result = tool
            .call(&json!({"path": "empty.txt", "content": ""}), &dummy_ctx())
            .await
            .unwrap();

        assert!(content_text(&result).contains("Created"));
        assert!(content_text(&result).contains("0 bytes"));
        assert!(content_text(&result).contains("0 lines"));
        assert!(dir.path().join("empty.txt").exists());
    }

    #[tokio::test]
    async fn test_name_and_definition() {
        let tool = WriteFileTool::new(PathBuf::from("/tmp"));
        assert_eq!(tool.name(), "write_file");

        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
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
