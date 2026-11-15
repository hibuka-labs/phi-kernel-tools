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
