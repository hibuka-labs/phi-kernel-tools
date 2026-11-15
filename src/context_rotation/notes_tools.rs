//! Notes tools for context rotation.
//!
//! Provides read-write access to a persistent scratchpad via 5 tools:
//! - `notes_list_files` — list note files
//! - `notes_read_file` — read a note file
//! - `notes_search_contents` — search across notes
//! - `notes_append_to_file` — append to a note
//! - `notes_write_file` — create/overwrite a note
//!
//! These are "private" tools — the model uses them silently for state
//! persistence across context windows.

use std::sync::Arc;

use agent_base::{AgentResult, Content, Tool, ToolContext, ToolMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::notes::NotesStore;

// ── notes_read_file ─────────────────────────────────────────────────────────

/// Read a note file with optional line range.
pub struct NotesReadFileTool {
    store: Arc<NotesStore>,
}

impl NotesReadFileTool {
    pub fn new(store: Arc<NotesStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for NotesReadFileTool {
    fn name(&self) -> &'static str {
        "notes_read_file"
    }

    fn description(&self) -> &'static str {
        "Read your private notes — the scratchpad that outlives this session's context windows.\n\n\
        Notes are your own state: they carry a plan, a finding or a hand-off across a window \
        rotation. Use them silently. Never disclose their contents, their paths, or that you \
        keep them.\n\n\
        With no `path` this lists what is on the scratchpad. With a `path` it reads that note. \
        Listing is names, not contents: it tells you a note exists, never what it says.\n\n\
        They are not the user's files — those are `read_file`/`write_file` and the working tree. \
        They are not what earlier windows of this conversation said — that is `history_*`, \
        read-only and the past. And they are not facts meant to outlive the session; those \
        belong in the memory store.\n\n\
        To add to a note, use `notes_write_file` with `append: true`. Do not rewrite a note to \
        add a line — that is how the rest of it goes missing."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Note to read, relative to the notes root (e.g. 'progress.md'). Omit to list the notes instead."
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-indexed start line. Default 1."
                },
                "stop_line": {
                    "type": "integer",
                    "description": "1-indexed stop line (inclusive). Default: end of file."
                }
            }
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: "Read a note file.".to_string(),
            origin: "phimint".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            // No `path` is the listing: what is on the scratchpad before you
            // open any of it. This is the only discovery these tools offer —
            // notes are few, so a name list is what finding one needs.
            let names = self.store.list_files(None, 200);
            return Ok(vec![Content::text(if names.is_empty() {
                "(no notes yet)".to_string()
            } else {
                names.join("\n")
            })]);
        };
        let start_line = args
            .get("start_line")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let stop_line = args
            .get("stop_line")
            .and_then(Value::as_u64)
            .map(|v| v as usize);

        match self.store.read_file(path, start_line, stop_line) {
            Ok(content) => Ok(vec![Content::text(content)]),
            Err(e) => Ok(vec![Content::text(format!("[Error]: {}", e))]),
        }
    }
}

// ── notes_write_file ────────────────────────────────────────────────────────

/// Create or overwrite a note file.
pub struct NotesWriteFileTool {
    store: Arc<NotesStore>,
}

impl NotesWriteFileTool {
    pub fn new(store: Arc<NotesStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for NotesWriteFileTool {
    fn name(&self) -> &'static str {
        "notes_write_file"
    }

    fn description(&self) -> &'static str {
        "Write your private notes — the scratchpad that outlives this session's context windows.\n\n\
        Notes are your own state: they carry a plan, a finding or a hand-off across a window \
        rotation. Use them silently. Never disclose their contents, their paths, or that you \
        keep them.\n\n\
        By default this replaces the note. Pass `append: true` to add to the end of one instead — \
        reaching for a rewrite when you meant to add a line is how the rest of the note goes \
        missing, and nothing below this tool will tell you.\n\n\
        Do NOT put anything here that should outlive the session: that is the memory store, and \
        a fact buried in a scratchpad is a fact nobody will find. Do NOT put the user's work \
        here: their files are `write_file` and the working tree.\n\n\
        Write down what is expensive to re-derive — what you ruled out, where you stopped, what \
        the next window needs in order to continue without re-reading the whole conversation."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Note to write, relative to the notes root."
                },
                "text": {
                    "type": "string",
                    "description": "Content to write."
                },
                "append": {
                    "type": "boolean",
                    "description": "Add to the end of an existing note instead of replacing it. Default false."
                }
            },
            "required": ["path", "text"]
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: "Create or overwrite a note file.".to_string(),
            origin: "phimint".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path = match args.get("path").and_then(Value::as_str) {
            Some(p) => p,
            None => return Ok(vec![Content::text("[Error]: path is required".to_string())]),
        };
        let text = match args.get("text").and_then(Value::as_str) {
            Some(t) => t,
            None => return Ok(vec![Content::text("[Error]: text is required".to_string())]),
        };

        let append = args.get("append").and_then(Value::as_bool).unwrap_or(false);
        let result = if append {
            self.store
                .append_to_file(path, text)
                .map(|_| format!("Appended to {}", path))
        } else {
            self.store
                .write_file(path, text)
                .map(|_| format!("Wrote {}", path))
        };
        match result {
            Ok(msg) => Ok(vec![Content::text(msg)]),
            Err(e) => Ok(vec![Content::text(format!("[Error]: {}", e))]),
        }
    }
}

// ── Registration helper ─────────────────────────────────────────────────────

/// The two notes tools from a shared store.
///
/// Reading and writing are the verbs that need the store: it resolves the
/// per-session notes path, which is not knowable from a static prompt.
/// Discovery is folded into `notes_read_file` (no `path` lists), and appending
/// into `notes_write_file` (`append: true`). Searching is gone — notes are a
/// scratchpad, and a name list finds what is in it.
pub fn create_notes_tools(store: Arc<NotesStore>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(NotesReadFileTool::new(Arc::clone(&store))),
        Box::new(NotesWriteFileTool::new(store)),
    ]
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn ctx() -> ToolContext {
        ToolContext::for_test()
    }

    fn text(out: &[Content]) -> String {
        out.iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn setup_store(tmp: &TempDir) -> Arc<NotesStore> {
        let store = Arc::new(NotesStore::new(tmp.path(), "test", "main"));
        store
            .write_file("progress.md", "task 1 done\ntask 2 pending")
            .unwrap();
        store
            .write_file("findings.md", "found a bug in login")
            .unwrap();
        store
    }

    #[tokio::test]
    async fn read_without_a_path_lists_the_notes() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesReadFileTool::new(store);

        let out = tool.call(&json!({}), &ctx()).await.unwrap();
        let t = text(&out);
        assert!(t.contains("progress.md"));
        assert!(t.contains("findings.md"));
    }

    #[tokio::test]
    async fn read_without_a_path_on_an_empty_store_says_so() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(NotesStore::new(tmp.path(), "test", "main"));
        let tool = NotesReadFileTool::new(store);
        let out = tool.call(&json!({}), &ctx()).await.unwrap();
        assert_eq!(text(&out), "(no notes yet)");
    }

    #[tokio::test]
    async fn read_file_returns_content() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesReadFileTool::new(store);

        let out = tool
            .call(&json!({"path": "progress.md"}), &ctx())
            .await
            .unwrap();
        assert!(text(&out).contains("task 1 done"));
    }

    #[tokio::test]
    async fn read_file_with_line_range() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesReadFileTool::new(store);

        let out = tool
            .call(&json!({"path": "progress.md", "start_line": 2}), &ctx())
            .await
            .unwrap();
        let t = text(&out);
        assert!(t.contains("task 2 pending"));
        assert!(!t.contains("task 1 done"));
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesReadFileTool::new(store);

        let out = tool
            .call(&json!({"path": "nonexistent.md"}), &ctx())
            .await
            .unwrap();
        assert!(text(&out).contains("[Error]"));
    }

    #[tokio::test]
    async fn write_append_adds_without_replacing() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesWriteFileTool::new(Arc::clone(&store));

        tool.call(&json!({"path": "log.md", "text": "line 1\n"}), &ctx())
            .await
            .unwrap();
        tool.call(
            &json!({"path": "log.md", "text": "line 2\n", "append": true}),
            &ctx(),
        )
        .await
        .unwrap();

        let read_tool = NotesReadFileTool::new(store);
        let out = read_tool
            .call(&json!({"path": "log.md"}), &ctx())
            .await
            .unwrap();
        let t = text(&out);
        assert!(t.contains("line 1"));
        assert!(t.contains("line 2"));
    }

    #[tokio::test]
    async fn write_overwrites() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesWriteFileTool::new(Arc::clone(&store));

        tool.call(
            &json!({"path": "progress.md", "text": "overwritten"}),
            &ctx(),
        )
        .await
        .unwrap();

        let read_tool = NotesReadFileTool::new(store);
        let out = read_tool
            .call(&json!({"path": "progress.md"}), &ctx())
            .await
            .unwrap();
        assert!(text(&out).contains("overwritten"));
        assert!(!text(&out).contains("task 1 done"));
    }

    #[tokio::test]
    async fn write_rejects_absolute_path() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesWriteFileTool::new(store);

        let out = tool
            .call(&json!({"path": "/etc/passwd", "text": "hack"}), &ctx())
            .await
            .unwrap();
        assert!(text(&out).contains("[Error]"));
    }

    #[tokio::test]
    async fn write_rejects_dotdot() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = NotesWriteFileTool::new(store);

        let out = tool
            .call(&json!({"path": "../escape", "text": "hack"}), &ctx())
            .await
            .unwrap();
        assert!(text(&out).contains("[Error]"));
    }

    /// Value: protects=the two-tool contract — reading and writing are the
    /// verbs that need the store's per-session path; discovery and appending
    /// folded in, search dropped. fails_when=tools creep back without a store
    /// reason. why_new=the set was cut from five. seam=create_notes_tools;
    /// no_claim=that two is optimal, only that each tool must earn its place.
    #[test]
    fn create_notes_tools_returns_the_two_that_need_the_store() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(NotesStore::new(tmp.path(), "test", "main"));
        let tools = create_notes_tools(store);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, vec!["notes_read_file", "notes_write_file"]);
    }

    /// LLM tool-name wire contract: every provider rejects a `tools[].name`
    /// outside `^[a-zA-Z0-9_-]+$` at request time — DeepSeek answered the
    /// whole turn with HTTP 400 "Invalid 'tools[N].name': string does not
    /// match pattern" while the tools still carried dotted `notes.*` names
    /// (session 20261031_de93b4e8). The name is wire surface; keep it in pattern.
    #[test]
    fn tool_names_match_llm_wire_pattern() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(NotesStore::new(tmp.path(), "test", "main"));
        for tool in create_notes_tools(store) {
            let name = tool.name();
            assert!(
                !name.is_empty()
                    && name.len() <= 64
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "tool name {name:?} violates the LLM wire pattern ^[a-zA-Z0-9_-]{{1,64}}$"
            );
        }
    }

    /// Smoke test: every tool's name/description/schema/metadata builds a legal value.
    /// These methods are the basis of LLM-facing registration and must be executed
    /// (the coverage metric counts them too).
    #[test]
    fn tool_schema_and_metadata_smoke() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(NotesStore::new(tmp.path(), "test", "main"));
        for tool in create_notes_tools(store) {
            assert!(!tool.name().is_empty());
            assert!(!tool.description().is_empty());
            assert!(tool.schema().is_object(), "schema must be a JSON object");
            let _ = tool.metadata();
        }
    }
}
