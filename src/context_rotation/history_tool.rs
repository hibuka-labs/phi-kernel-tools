//! `history` — one entry point over the archived context windows.
//!
//! Four verbs — list windows, list items, read one, search — are one question
//! asked four ways: *what is in my past?* The store underneath is structured
//! (windows → items), so this cannot be plain files the way notes can, but it
//! also does not need four tool names competing for the model's attention.
//!
//! The per-op formatting lives in the four inner tools and is unchanged; this
//! is the front door, not a rewrite.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use agent_base::{Content, Tool, ToolContext, ToolMetadata};

use super::history::HistoryStore;
use super::history_tools::{
    HistoryListItemsTool, HistoryListWindowsTool, HistoryReadItemTool, HistorySearchContentsTool,
};

/// One tool, four operations, over the archived context windows.
pub struct HistoryTool {
    windows: HistoryListWindowsTool,
    items: HistoryListItemsTool,
    read: HistoryReadItemTool,
    search: HistorySearchContentsTool,
}

impl HistoryTool {
    pub fn new(store: Arc<HistoryStore>) -> Self {
        Self {
            windows: HistoryListWindowsTool::new(Arc::clone(&store)),
            items: HistoryListItemsTool::new(Arc::clone(&store)),
            read: HistoryReadItemTool::new(Arc::clone(&store)),
            search: HistorySearchContentsTool::new(store),
        }
    }

    /// The `op` values, for the error a wrong guess deserves.
    const OPS: [&'static str; 4] = ["list_windows", "list_items", "read", "search"];

    /// Arguments for the inner tool: everything the caller sent except `op`.
    fn inner_args(args: &Value) -> Value {
        let mut out = args.as_object().cloned().unwrap_or_default();
        out.remove("op");
        Value::Object(out)
    }
}

#[async_trait]
impl Tool for HistoryTool {
    fn name(&self) -> &'static str {
        "history"
    }

    fn description(&self) -> &'static str {
        "Read your own past: the context windows this conversation has already rotated out of.\n\n\
        **Looking for something this conversation already said or did? This is the tool.** A \
        claim you want to check you made, a file a tool showed you earlier, a decision that has \
        gone missing, wording you want verbatim — all of it is here. Not in `notes_*` (that is \
        what you wrote going forward), and not in a grep over the working tree (that is the \
        codebase's record, not the conversation's).\n\n\
        A window that rotated is archived, not summarised: what you said and what the tools \
        returned is still there verbatim. In a long session that is several windows — search \
        first, read only what matches.\n\n\
        Four `op` values answer four shapes of the same question:\n\
        - `list_windows` — what is out there to look at.\n\
        - `list_items` — what a window (or all of them) contains, optionally by role or tool.\n\
        - `read` — one message in full. Summaries are previews; this is the text.\n\
        - `search` — where a phrase appears across the archive.\n\n\
        Do NOT use it to write — this is the past and it is read-only. Notes you are carrying \
        forward go in `notes_write_file`. Do NOT use it to find something in the working tree: \
        that is `search_content`/`read_file`, and reading the archive to find a source file is \
        the wrong half of the record.\n\n\
        An archived window is not a transcript of the present: it stops where the window ended. \
        And `list_items` returns previews — a preview tells you a message exists, not what it \
        said, so `read` before you quote it."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": Self::OPS,
                    "description": "What to do. `list_windows` needs nothing else; `list_items` takes window_id/role/tool_name; `read` needs window_id and item_id; `search` needs query."
                },
                "window_id": { "type": "integer", "description": "Window to work in. Omit where the op allows all windows." },
                "item_id": { "type": "integer", "description": "Message index within the window (`read`)." },
                "role": { "type": "string", "description": "Filter by message role: user, assistant, tool, system." },
                "tool_name": { "type": "string", "description": "Filter tool messages by tool name." },
                "query": { "type": "string", "description": "Text to look for (`search`)." },
                "limit": { "type": "integer", "description": "Max rows to return." },
                "recent_first": { "type": "boolean", "description": "Most recent first." },
                "offset_chars": { "type": "integer", "description": "Continue a long message (`read`)." },
                "limit_chars": { "type": "integer", "description": "Cap a long message (`read`)." }
            },
            "required": ["op"]
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: "Read the archived context windows.".to_string(),
            origin: "phimint".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> agent_base::AgentResult<Vec<Content>> {
        let Some(op) = args.get("op").and_then(Value::as_str) else {
            return Ok(vec![Content::text(format!(
                "[Error]: `op` is required — one of {}",
                Self::OPS.join(", ")
            ))]);
        };
        let inner = Self::inner_args(args);
        match op {
            "list_windows" => self.windows.call(&inner, ctx).await,
            "list_items" => self.items.call(&inner, ctx).await,
            "read" => self.read.call(&inner, ctx).await,
            "search" => self.search.call(&inner, ctx).await,
            other => Ok(vec![Content::text(format!(
                "[Error]: unknown op `{other}` — one of {}",
                Self::OPS.join(", ")
            ))]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::ChatMessage;
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

    fn setup_store(tmp: &TempDir) -> Arc<HistoryStore> {
        let store = Arc::new(HistoryStore::new(tmp.path(), "test"));
        store
            .archive_window(
                1,
                &[
                    ChatMessage::user("fix the login bug"),
                    ChatMessage::assistant("I'll investigate"),
                    ChatMessage::tool_with_name("tc1", "shell", "error: connection refused"),
                ],
            )
            .unwrap();
        store
    }

    /// Value: protects=the one-name contract — four verbs over one structured
    /// store is one question asked four ways, and four names cost the model a
    /// choice every turn for no gain. fails_when=the four verbs stop
    /// answering, or the dispatcher reaches for a tool the model cannot name.
    /// why_new=the set was collapsed into one entry point.
    /// seam=the `op` dispatch; no_claim=that one name is optimal, only that
    /// each verb must answer its own shape.
    #[tokio::test]
    async fn one_tool_answers_all_four_verbs() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = HistoryTool::new(store);

        // list_windows — what is out there to look at. It lists windows, not
        // their contents: the count is what tells the model there is a window.
        let out = tool
            .call(&json!({"op": "list_windows"}), &ctx())
            .await
            .unwrap();
        let t = text(&out);
        assert!(
            t.contains("Archived context windows") && t.contains("3 messages"),
            "list_windows must list the archive: {t}"
        );

        // list_items — what a window contains.
        let out = tool
            .call(&json!({"op": "list_items", "window_id": 1}), &ctx())
            .await
            .unwrap();
        let t = text(&out);
        assert!(
            t.contains("fix the login bug"),
            "list_items must list the window's messages: {t}"
        );

        // read — one message in full.
        let out = tool
            .call(&json!({"op": "read", "window_id": 1, "item_id": 0}), &ctx())
            .await
            .unwrap();
        let t = text(&out);
        assert!(
            t.contains("fix the login bug"),
            "read must return the message text: {t}"
        );

        // search — where a phrase appears across the archive.
        let out = tool
            .call(&json!({"op": "search", "query": "login"}), &ctx())
            .await
            .unwrap();
        let t = text(&out);
        assert!(t.contains("login"), "search must find the phrase: {t}");
    }

    /// The `op` is the whole contract: a caller that omits it, or names one
    /// that does not exist, must be told which verbs exist rather than getting
    /// a silent empty answer.
    #[tokio::test]
    async fn a_missing_or_unknown_op_names_the_verbs() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = HistoryTool::new(store);

        let out = tool.call(&json!({}), &ctx()).await.unwrap();
        let t = text(&out);
        assert!(t.contains("op"), "a missing op must be reported: {t}");
        assert!(
            t.contains("list_windows") && t.contains("search"),
            "the error must name the verbs that do work: {t}"
        );

        let out = tool
            .call(&json!({"op": "list_everything"}), &ctx())
            .await
            .unwrap();
        let t = text(&out);
        assert!(
            t.contains("unknown op"),
            "an unknown op must be reported: {t}"
        );
    }

    /// The four verbs are still `Tool`s — they carry their own names and
    /// schemas, and `HistoryTool` dispatches to them. The dispatcher is the
    /// only caller now, so the inner tools' own metadata is reachable only
    /// through it; asserting it here keeps the wire contract honest (a verb
    /// whose schema lost a required field is a verb the model cannot call).
    #[test]
    fn the_four_verbs_keep_their_wire_contract() {
        let tmp = TempDir::new().unwrap();
        let store = setup_store(&tmp);
        let tool = HistoryTool::new(store);

        let names = [
            tool.windows.name(),
            tool.items.name(),
            tool.read.name(),
            tool.search.name(),
        ];
        assert_eq!(
            names,
            [
                "history_list_windows",
                "history_list_items",
                "history_read_item",
                "history_search_contents"
            ]
        );

        for (verb, name, schema, description, metadata) in [
            (
                "list_windows",
                tool.windows.name(),
                tool.windows.schema(),
                tool.windows.description(),
                tool.windows.metadata(),
            ),
            (
                "list_items",
                tool.items.name(),
                tool.items.schema(),
                tool.items.description(),
                tool.items.metadata(),
            ),
            (
                "read",
                tool.read.name(),
                tool.read.schema(),
                tool.read.description(),
                tool.read.metadata(),
            ),
            (
                "search",
                tool.search.name(),
                tool.search.schema(),
                tool.search.description(),
                tool.search.metadata(),
            ),
        ] {
            assert!(
                schema.get("properties").is_some(),
                "{verb}: the verb must expose its arguments"
            );
            assert!(
                !description.is_empty(),
                "{verb}: the verb must say what it does"
            );
            assert_eq!(
                metadata.name, name,
                "{verb}: the metadata must name the verb"
            );
        }

        // `read` needs both coordinates; `search` needs its phrase.
        let read_schema = tool.read.schema();
        let read_required = read_schema["required"].as_array().unwrap();
        assert!(
            read_required.iter().any(|v| v == "window_id")
                && read_required.iter().any(|v| v == "item_id"),
            "read must require both coordinates: {read_required:?}"
        );
        let search_schema = tool.search.schema();
        let search_required = search_schema["required"].as_array().unwrap();
        assert!(
            search_required.iter().any(|v| v == "query"),
            "search must require its phrase: {search_required:?}"
        );
    }

    /// `op` is the dispatch key and nothing else — every other argument
    /// belongs to the verb. A `window_id` that leaked into the inner call
    /// would be silently ignored, so the inner arguments must be exactly the
    /// caller's minus `op`.
    #[test]
    fn inner_args_drops_only_op() {
        let args = json!({
            "op": "search",
            "query": "login",
            "window_id": 3,
            "limit": 5
        });
        let inner = HistoryTool::inner_args(&args);
        assert!(inner.get("op").is_none(), "op must not reach the verb");
        assert_eq!(inner.get("query").unwrap(), "login");
        assert_eq!(inner.get("window_id").unwrap(), 3);
        assert_eq!(inner.get("limit").unwrap(), 5);
    }
}
