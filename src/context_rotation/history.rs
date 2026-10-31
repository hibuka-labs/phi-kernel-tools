//! History storage for context rotation.
//!
//! Archives context window messages to JSONL files and maintains
//! per-session metadata. Used by the rotation policy during
//! window resets.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use agent_base::ChatMessage;

// ── Metadata ────────────────────────────────────────────────────────────────

/// Metadata for a single context window.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowMeta {
    /// Window ID (zero-padded 3-digit string, e.g. "001").
    pub id: String,
    /// ISO 8601 timestamp when the window was archived.
    pub started_at: String,
    /// Number of messages archived in this window.
    pub message_count: usize,
}

/// Session-level metadata stored in `metadata.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub windows: Vec<WindowMeta>,
}

// ── ChatMessage JSONL serialization ─────────────────────────────────────────

/// A single JSONL record for an archived chat message.
///
/// The flat fields (`role` / `name` / `content`) are the *query* view used by
/// `list_items` / `search_contents` / `read_item`. `msg` is the lossless
/// source — the full `ChatMessage` as the engine knew it, `tool_calls` and
/// `tool_call_id` included — so an archived window can be replayed and
/// restored, not just browsed. Records written before `msg` existed simply
/// lack it; [`MessageRecord::into_message`] falls back to the flat fields
/// (text only — tool call structure is already gone in those files).
#[derive(Serialize, Deserialize)]
struct MessageRecord {
    window: usize,
    index: usize,
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    content: String,
    ts: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    msg: Option<serde_json::Value>,
}

impl MessageRecord {
    /// Rebuild the engine-level message this record was written from.
    fn into_message(self) -> ChatMessage {
        if let Some(value) = self.msg {
            match serde_json::from_value::<ChatMessage>(value) {
                Ok(msg) => return msg,
                Err(e) => {
                    tracing::warn!(error = %e, "archived msg payload failed to decode; falling back to flat fields");
                }
            }
        }
        // Legacy record: flat fields only. Text survives, tool call pairing
        // does not — those archives were written before `msg` existed.
        match self.role.as_str() {
            "system" => ChatMessage::System {
                content: self.content,
                ephemeral: false,
            },
            "user" => ChatMessage::User {
                content: self.content,
                images: Vec::new(),
                ephemeral: false,
            },
            "assistant" => ChatMessage::Assistant {
                content: if self.content.is_empty() {
                    None
                } else {
                    Some(self.content)
                },
                reasoning_content: None,
                thinking_signature: None,
                tool_calls: None,
            },
            "tool" => ChatMessage::Tool {
                // Synthetic id: the original pairing is gone. Callers that
                // need a valid sequence must run [`sanitize_for_resume`].
                tool_call_id: format!("archived-{}-{}", self.window, self.index),
                name: self.name,
                content: self.content,
                // The archive line carries words only; pictures a tool result
                // once held are gone by the time it is rotated out.
                images: Vec::new(),
            },
            role => ChatMessage::Custom {
                role: role.to_string(),
                data: serde_json::json!({ "content": self.content }),
            },
        }
    }
}

fn message_role(msg: &ChatMessage) -> String {
    match msg {
        ChatMessage::System { .. } => "system".to_string(),
        ChatMessage::User { .. } => "user".to_string(),
        ChatMessage::Assistant { .. } => "assistant".to_string(),
        ChatMessage::Tool { .. } => "tool".to_string(),
        ChatMessage::Custom { role, .. } => role.clone(),
    }
}

fn message_content(msg: &ChatMessage) -> String {
    match msg {
        ChatMessage::System { content, .. } => content.clone(),
        ChatMessage::User { content, .. } => content.clone(),
        ChatMessage::Assistant { content, .. } => content.clone().unwrap_or_default(),
        ChatMessage::Tool { content, .. } => content.clone(),
        ChatMessage::Custom { data, .. } => data.to_string(),
    }
}

fn message_name(msg: &ChatMessage) -> Option<String> {
    match msg {
        ChatMessage::Tool { name, .. } => name.clone(),
        _ => None,
    }
}

// ── Query result types ──────────────────────────────────────────────────────

/// Summary of a message item for `list_items`.
#[derive(Clone, Debug)]
pub struct ItemSummary {
    pub window: usize,
    pub index: usize,
    pub role: String,
    pub name: Option<String>,
    pub content_preview: String,
}

/// Search result for `search_contents`.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub window: usize,
    pub index: usize,
    pub role: String,
    pub name: Option<String>,
    pub match_preview: String,
}

// ── Helper functions ────────────────────────────────────────────────────────

fn truncate_str(s: &str, max_chars: usize) -> String {
    // Characters, not bytes: `&s[..max_chars]` is a byte slice and panics when
    // it lands mid-sequence inside a multi-byte char (archived history is
    // largely CJK, 3 bytes per char).
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max_chars).collect::<String>())
    }
}

/// Extract a preview around the first match of `query` in `content`.
///
/// All offsets applied to `content` are CHARACTER offsets, for the same
/// reason as [`truncate_str`]. `pos` from `find` is a byte index into the
/// lowercased copy (whose byte layout may even differ from the original —
/// e.g. 'İ' lowercases to two chars), so it is first mapped to a char index
/// before touching the original string.
fn extract_match_preview(content: &str, query_lower: &str, context_chars: usize) -> String {
    let content_lower = content.to_lowercase();
    if let Some(pos) = content_lower.find(query_lower) {
        let match_char_idx = content_lower[..pos].chars().count();
        let query_chars = query_lower.chars().count();
        let start = match_char_idx.saturating_sub(context_chars);
        let end = match_char_idx + query_chars + context_chars;
        let snippet: String = content.chars().skip(start).take(end - start).collect();
        if start > 0 {
            format!("...{}...", snippet)
        } else {
            format!("{}...", snippet)
        }
    } else {
        truncate_str(content, context_chars * 2)
    }
}

// ── HistoryStore ────────────────────────────────────────────────────────────

/// Manages history storage at `~/.phimint/history/<session_id>/`.
pub struct HistoryStore {
    session_dir: PathBuf,
}

impl HistoryStore {
    /// Create a new store for the given base directory and session ID.
    ///
    /// `base_dir` is typically `~/.phimint/`. The session directory
    /// (`<base_dir>/history/<session_id>/`) is created on first write.
    pub fn new(base_dir: &Path, session_id: &str) -> Self {
        Self {
            session_dir: base_dir.join("history").join(session_id),
        }
    }

    /// Archive messages to a JSONL file for the given window.
    ///
    /// Creates `<session_dir>/windows/<window_id>.jsonl` and updates
    /// `metadata.json`. Returns the number of messages written.
    pub fn archive_window(
        &self,
        window_id: usize,
        messages: &[ChatMessage],
    ) -> std::io::Result<usize> {
        // Ensure directories exist
        let windows_dir = self.session_dir.join("windows");
        fs::create_dir_all(&windows_dir)?;

        // Write JSONL
        let file_name = format!("{:03}.jsonl", window_id);
        let file_path = windows_dir.join(&file_name);
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)?;

        let ts = chrono::Utc::now().to_rfc3339();
        for (i, msg) in messages.iter().enumerate() {
            let record = MessageRecord {
                window: window_id,
                index: i,
                role: message_role(msg),
                name: message_name(msg),
                content: message_content(msg),
                ts: ts.clone(),
                // Lossless copy: keeps `tool_calls` / `tool_call_id` so the
                // window can be replayed and restored, not just searched.
                // A serialization failure degrades to the flat fields rather
                // than losing the archive.
                msg: serde_json::to_value(msg).ok(),
            };
            writeln!(file, "{}", serde_json::to_string(&record)?)?;
        }
        file.flush()?;

        // Update metadata
        self.update_metadata(window_id, messages.len())?;

        Ok(messages.len())
    }

    /// Read the session metadata, or create a default if it doesn't exist.
    pub fn load_metadata(&self) -> SessionMeta {
        let meta_path = self.session_dir.join("metadata.json");
        match fs::read_to_string(&meta_path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_else(|_| SessionMeta {
                session_id: self.session_id().to_string(),
                windows: Vec::new(),
            }),
            Err(_) => SessionMeta {
                session_id: self.session_id().to_string(),
                windows: Vec::new(),
            },
        }
    }

    fn update_metadata(&self, window_id: usize, message_count: usize) -> std::io::Result<()> {
        let mut meta = self.load_metadata();
        let window_id_str = format!("{:03}", window_id);

        // Upsert window entry
        if let Some(w) = meta.windows.iter_mut().find(|w| w.id == window_id_str) {
            w.message_count = message_count;
            w.started_at = chrono::Utc::now().to_rfc3339();
        } else {
            meta.windows.push(WindowMeta {
                id: window_id_str,
                started_at: chrono::Utc::now().to_rfc3339(),
                message_count,
            });
        }

        let meta_path = self.session_dir.join("metadata.json");
        fs::create_dir_all(&self.session_dir)?;
        fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    }

    fn session_id(&self) -> &str {
        self.session_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
    }

    /// List archived window IDs (sorted ascending).
    pub fn list_windows(&self) -> Vec<String> {
        let meta = self.load_metadata();
        let mut ids: Vec<String> = meta.windows.iter().map(|w| w.id.clone()).collect();
        ids.sort();
        ids
    }

    /// Read all records from a window's JSONL file.
    fn read_records(&self, window_id: usize) -> std::io::Result<Vec<MessageRecord>> {
        let file_name = format!("{:03}.jsonl", window_id);
        let path = self.session_dir.join("windows").join(&file_name);
        let content = fs::read_to_string(&path)?;
        let records: Vec<MessageRecord> = content
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        Ok(records)
    }

    /// Rebuild one archived window as engine-level [`ChatMessage`]s.
    ///
    /// Lossless when the records carry the `msg` payload (including
    /// `tool_calls`); text-only for archives written before that field
    /// existed.
    pub fn load_window_messages(&self, window_id: usize) -> Vec<ChatMessage> {
        match self.read_records(window_id) {
            Ok(records) => records
                .into_iter()
                .map(MessageRecord::into_message)
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Rebuild every archived window, oldest first.
    ///
    /// This is the display/recovery path for a session's full history — the
    /// live `messages.jsonl` only ever holds the *current* window.
    pub fn load_all_messages(&self) -> Vec<ChatMessage> {
        let mut ids: Vec<usize> = self
            .list_windows()
            .iter()
            .filter_map(|id| id.parse::<usize>().ok())
            .collect();
        ids.sort_unstable();
        ids.into_iter()
            .flat_map(|id| self.load_window_messages(id))
            .collect()
    }

    /// List message items in a window with optional filtering.
    ///
    /// Returns a summary of each matching message (role, name, content preview).
    pub fn list_items(
        &self,
        window_id: Option<usize>,
        role: Option<&str>,
        tool_name: Option<&str>,
        limit: usize,
        recent_first: bool,
    ) -> Vec<ItemSummary> {
        let windows = match window_id {
            Some(id) => vec![id],
            None => {
                let ids = self.list_windows();
                ids.iter()
                    .filter_map(|id| id.parse::<usize>().ok())
                    .collect()
            }
        };

        let mut items = Vec::new();
        for wid in windows {
            let records = match self.read_records(wid) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for rec in records {
                if let Some(r) = role
                    && rec.role != r
                {
                    continue;
                }
                if let Some(tn) = tool_name {
                    match &rec.name {
                        Some(n) if n == tn => {}
                        _ => continue,
                    }
                }
                items.push(ItemSummary {
                    window: rec.window,
                    index: rec.index,
                    role: rec.role,
                    name: rec.name,
                    content_preview: truncate_str(&rec.content, 200),
                });
            }
        }

        if recent_first {
            items.reverse();
        }
        items.truncate(limit);
        items
    }

    /// Read a single item's content by window_id and item index.
    ///
    /// Supports optional character range (offset_chars, limit_chars).
    pub fn read_item(
        &self,
        window_id: usize,
        item_id: usize,
        offset_chars: Option<usize>,
        limit_chars: Option<usize>,
    ) -> Option<(String, usize)> {
        let records = self.read_records(window_id).ok()?;
        let rec = records.into_iter().find(|r| r.index == item_id)?;

        let content = rec.content;
        // Characters, not bytes: archived messages are largely CJK, and a byte
        // range that lands mid-sequence panics. The parameters document a
        // character range, so count characters end to end (including `total`,
        // which callers print as `[chars a..b/total]`).
        let total_len = content.chars().count();
        let start = offset_chars.unwrap_or(0).min(total_len);
        let end = match limit_chars {
            Some(limit) => (start + limit).min(total_len),
            None => total_len,
        };
        let slice: String = content.chars().skip(start).take(end - start).collect();
        Some((slice, total_len))
    }

    /// Search content across windows.
    ///
    /// Returns matching items with content preview.
    pub fn search_contents(
        &self,
        query: &str,
        window_id: Option<usize>,
        role: Option<&str>,
        limit: usize,
    ) -> Vec<SearchResult> {
        let windows = match window_id {
            Some(id) => vec![id],
            None => {
                let ids = self.list_windows();
                ids.iter()
                    .filter_map(|id| id.parse::<usize>().ok())
                    .collect()
            }
        };

        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for wid in windows {
            let records = match self.read_records(wid) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for rec in records {
                if let Some(r) = role
                    && rec.role != r
                {
                    continue;
                }
                if !rec.content.to_lowercase().contains(&query_lower) {
                    continue;
                }
                // Find the match context
                let preview = extract_match_preview(&rec.content, &query_lower, 100);
                results.push(SearchResult {
                    window: rec.window,
                    index: rec.index,
                    role: rec.role,
                    name: rec.name,
                    match_preview: preview,
                });
                if results.len() >= limit {
                    return results;
                }
            }
        }

        results
    }
}

// ── Thread Hint ─────────────────────────────────────────────────────────────

/// Maximum size for thread hint injection (4 KB).
const THREAD_HINT_MAX_BYTES: usize = 4096;

/// Read the thread hint from notes, if it exists and is within size limit.
///
/// Path: `~/.phimint/notes/<session_id>/<agent_name>/thread_hint.md`
pub fn read_thread_hint(base_dir: &Path, session_id: &str, agent_name: &str) -> Option<String> {
    let path = base_dir
        .join("notes")
        .join(session_id)
        .join(agent_name)
        .join("thread_hint.md");

    let content = fs::read_to_string(&path).ok()?;
    if content.len() > THREAD_HINT_MAX_BYTES {
        tracing::warn!(
            path = %path.display(),
            size = content.len(),
            max = THREAD_HINT_MAX_BYTES,
            "thread hint exceeds size limit, skipping"
        );
        return None;
    }
    Some(content)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_messages(n: usize) -> Vec<ChatMessage> {
        (0..n)
            .map(|i| match i % 3 {
                0 => ChatMessage::user(format!("user message {}", i)),
                1 => ChatMessage::assistant(format!("assistant message {}", i)),
                _ => ChatMessage::tool(format!("tc_{}", i), format!("tool output {}", i)),
            })
            .collect()
    }

    #[test]
    fn archive_creates_jsonl_file() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        let msgs = make_messages(5);
        store.archive_window(1, &msgs).unwrap();

        let jsonl_path = tmp
            .path()
            .join("history")
            .join("test_session")
            .join("windows")
            .join("001.jsonl");
        assert!(jsonl_path.exists());
    }

    #[test]
    fn archive_jsonl_format_correct() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        let msgs = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi there"),
            ChatMessage::tool_with_name("tc1", "shell", "output"),
        ];
        store.archive_window(1, &msgs).unwrap();

        let jsonl_path = tmp
            .path()
            .join("history")
            .join("test_session")
            .join("windows")
            .join("001.jsonl");
        let content = fs::read_to_string(&jsonl_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);

        // Check first record
        let rec: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(rec["window"], 1);
        assert_eq!(rec["index"], 0);
        assert_eq!(rec["role"], "user");
        assert_eq!(rec["content"], "hello");
        assert!(rec["ts"].as_str().is_some());

        // Check tool record has name
        let tool_rec: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(tool_rec["role"], "tool");
        assert_eq!(tool_rec["name"], "shell");
    }

    #[test]
    fn archive_updates_metadata() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        store.archive_window(1, &make_messages(3)).unwrap();
        store.archive_window(2, &make_messages(5)).unwrap();

        let meta = store.load_metadata();
        assert_eq!(meta.session_id, "test_session");
        assert_eq!(meta.windows.len(), 2);
        assert_eq!(meta.windows[0].id, "001");
        assert_eq!(meta.windows[0].message_count, 3);
        assert_eq!(meta.windows[1].id, "002");
        assert_eq!(meta.windows[1].message_count, 5);
    }

    #[test]
    fn archive_overwrites_existing_window() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        store.archive_window(1, &make_messages(3)).unwrap();
        store.archive_window(1, &make_messages(7)).unwrap();

        let meta = store.load_metadata();
        assert_eq!(meta.windows.len(), 1);
        assert_eq!(meta.windows[0].message_count, 7);
    }

    #[test]
    fn list_windows_returns_sorted_ids() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        store.archive_window(2, &make_messages(1)).unwrap();
        store.archive_window(1, &make_messages(1)).unwrap();
        store.archive_window(3, &make_messages(1)).unwrap();

        let ids = store.list_windows();
        assert_eq!(ids, vec!["001", "002", "003"]);
    }

    #[test]
    fn load_metadata_returns_default_when_missing() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "new_session");
        let meta = store.load_metadata();
        assert_eq!(meta.session_id, "new_session");
        assert!(meta.windows.is_empty());
    }

    #[test]
    fn archive_empty_messages() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        let count = store.archive_window(1, &[]).unwrap();
        assert_eq!(count, 0);

        let meta = store.load_metadata();
        assert_eq!(meta.windows[0].message_count, 0);
    }

    #[test]
    fn read_thread_hint_returns_content() {
        let tmp = TempDir::new().unwrap();
        let hint_dir = tmp.path().join("notes").join("s1").join("agent");
        fs::create_dir_all(&hint_dir).unwrap();
        fs::write(hint_dir.join("thread_hint.md"), "previous context").unwrap();

        let hint = read_thread_hint(tmp.path(), "s1", "agent");
        assert_eq!(hint.as_deref(), Some("previous context"));
    }

    #[test]
    fn read_thread_hint_returns_none_when_missing() {
        let tmp = TempDir::new().unwrap();
        let hint = read_thread_hint(tmp.path(), "s1", "agent");
        assert!(hint.is_none());
    }

    #[test]
    fn read_thread_hint_skips_oversized() {
        let tmp = TempDir::new().unwrap();
        let hint_dir = tmp.path().join("notes").join("s1").join("agent");
        fs::create_dir_all(&hint_dir).unwrap();
        // Write 5KB (exceeds 4KB limit)
        fs::write(hint_dir.join("thread_hint.md"), "x".repeat(5000)).unwrap();

        let hint = read_thread_hint(tmp.path(), "s1", "agent");
        assert!(hint.is_none());
    }

    #[test]
    fn read_thread_hint_accepts_exactly_4kb() {
        let tmp = TempDir::new().unwrap();
        let hint_dir = tmp.path().join("notes").join("s1").join("agent");
        fs::create_dir_all(&hint_dir).unwrap();
        fs::write(
            hint_dir.join("thread_hint.md"),
            "x".repeat(THREAD_HINT_MAX_BYTES),
        )
        .unwrap();

        let hint = read_thread_hint(tmp.path(), "s1", "agent");
        assert!(hint.is_some());
    }

    // ── list_items tests ──

    #[test]
    fn list_items_returns_all_messages() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store.archive_window(1, &make_messages(5)).unwrap();

        let items = store.list_items(None, None, None, 100, false);
        assert_eq!(items.len(), 5);
        assert_eq!(items[0].role, "user");
        assert_eq!(items[1].role, "assistant");
        assert_eq!(items[2].role, "tool");
    }

    #[test]
    fn list_items_filters_by_role() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store.archive_window(1, &make_messages(6)).unwrap();

        let users = store.list_items(None, Some("user"), None, 100, false);
        assert!(users.iter().all(|i| i.role == "user"));
        assert_eq!(users.len(), 2); // indices 0, 3
    }

    #[test]
    fn list_items_filters_by_window() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store.archive_window(1, &make_messages(3)).unwrap();
        store.archive_window(2, &make_messages(3)).unwrap();

        let items = store.list_items(Some(1), None, None, 100, false);
        assert_eq!(items.len(), 3);
        assert!(items.iter().all(|i| i.window == 1));
    }

    #[test]
    fn list_items_respects_limit() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store.archive_window(1, &make_messages(10)).unwrap();

        let items = store.list_items(None, None, None, 3, false);
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn list_items_recent_first() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store.archive_window(1, &make_messages(3)).unwrap();

        let items = store.list_items(None, None, None, 100, true);
        assert_eq!(items[0].index, 2); // last message first
        assert_eq!(items[2].index, 0); // first message last
    }

    // ── read_item tests ──

    #[test]
    fn read_item_returns_content() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(1, &[ChatMessage::user("hello world")])
            .unwrap();

        let (content, total_len) = store.read_item(1, 0, None, None).unwrap();
        assert_eq!(content, "hello world");
        assert_eq!(total_len, 11);
    }

    #[test]
    fn read_item_with_offset_and_limit() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(1, &[ChatMessage::user("hello world")])
            .unwrap();

        let (content, total_len) = store.read_item(1, 0, Some(6), Some(5)).unwrap();
        assert_eq!(content, "world");
        assert_eq!(total_len, 11);
    }

    #[test]
    fn read_item_offset_and_limit_count_characters() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(1, &[ChatMessage::user("记忆文件测试")])
            .unwrap();

        // 6 chars / 18 bytes: a byte offset of 2 would land mid-sequence and
        // panic, so the range must be honoured in characters.
        let (content, total_len) = store.read_item(1, 0, Some(2), Some(3)).unwrap();
        assert_eq!(content, "文件测");
        assert_eq!(total_len, 6);
    }

    #[test]
    fn read_item_returns_none_for_missing() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        assert!(store.read_item(99, 0, None, None).is_none());
    }

    // ── search_contents tests ──

    #[test]
    fn search_contents_finds_match() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(
                1,
                &[
                    ChatMessage::user("fix the login bug"),
                    ChatMessage::assistant("I'll investigate"),
                ],
            )
            .unwrap();

        let results = store.search_contents("login", None, None, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].window, 1);
        assert_eq!(results[0].index, 0);
        assert!(results[0].match_preview.contains("login"));
    }

    #[test]
    fn search_contents_case_insensitive() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(1, &[ChatMessage::user("Hello World")])
            .unwrap();

        let results = store.search_contents("hello", None, None, 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_contents_filters_by_role() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(
                1,
                &[
                    ChatMessage::user("error in code"),
                    ChatMessage::assistant("no error here"),
                ],
            )
            .unwrap();

        let results = store.search_contents("error", None, Some("user"), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].role, "user");
    }

    #[test]
    fn search_contents_respects_limit() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(
                1,
                &[
                    ChatMessage::user("test message"),
                    ChatMessage::assistant("test again"),
                    ChatMessage::user("test third"),
                ],
            )
            .unwrap();

        let results = store.search_contents("test", None, None, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_contents_across_windows() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(1, &[ChatMessage::user("needle")])
            .unwrap();
        store
            .archive_window(2, &[ChatMessage::assistant("needle too")])
            .unwrap();

        let results = store.search_contents("needle", None, None, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_contents_cjk_match_does_not_panic() {
        // Regression: extract_match_preview used to compute byte offsets from
        // `context_chars` and slice `&content[start..end]`, panicking with
        // "byte index N is not a char boundary" whenever the 100-char context
        // window landed inside a CJK char (3 bytes each).
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        // >100 CJK chars before the match so the window start lands mid-text,
        // and enough trailing CJK chars that the end lands mid-char too.
        let content = format!("{}Step 0-4{}", "验".repeat(150), "证".repeat(150),);
        store
            .archive_window(1, &[ChatMessage::user(&content)])
            .unwrap();

        let results = store.search_contents("Step 0-4", None, None, 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].match_preview.contains("Step 0-4"));
        assert!(results[0].match_preview.starts_with("..."));
    }

    #[test]
    fn search_contents_no_match_cjk_content_does_not_panic() {
        // The no-match fallback truncates to context_chars * 2 — the same
        // byte-slice panic lived there via truncate_str.
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(1, &[ChatMessage::user("中".repeat(500))])
            .unwrap();

        // Force the fallback path by calling the helper through a query that
        // IS present per the `contains` pre-filter but exercises preview on
        // CJK: use a query that exists so search returns; the fallback is
        // covered directly below.
        let results = store.search_contents("中", None, None, 10);
        assert_eq!(results.len(), 1);

        // Direct fallback coverage: preview of CJK content with no match.
        let preview = extract_match_preview(&"文".repeat(500), "zzz", 100);
        assert_eq!(preview.chars().count(), 200 + 3); // 200 chars + "..."
    }

    #[test]
    fn list_items_cjk_preview_does_not_panic() {
        // list_items previews go through truncate_str(&content, 200) — a byte
        // slice at 200 panics on CJK content longer than 200 bytes.
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test");
        store
            .archive_window(1, &[ChatMessage::user("汉".repeat(300))])
            .unwrap();

        let items = store.list_items(None, None, None, 10, false);
        assert_eq!(items.len(), 1);
        // 200 chars + "..." suffix
        assert!(items[0].content_preview.ends_with("..."));
        assert_eq!(items[0].content_preview.chars().count(), 203);
    }

    // ── load_all_messages: archives must be restorable, not just searchable ──

    #[test]
    fn load_all_messages_round_trips_tool_calls() {
        // The archive used to drop `tool_calls`, so a replayed window lost
        // every call and only kept the text. Restoring a session needs the
        // structure back or `validate_message_sequence` rejects the result.
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        let msgs = vec![
            ChatMessage::user("list the files"),
            ChatMessage::Assistant {
                content: Some("looking".into()),
                reasoning_content: Some("stale".into()),
                thinking_signature: Some("sig".into()),
                tool_calls: Some(vec![agent_base::ToolCallMessage {
                    id: "tc_1".into(),
                    name: "shell".into(),
                    arguments: "{\"cmd\":\"ls\"}".into(),
                }]),
            },
            ChatMessage::Tool {
                tool_call_id: "tc_1".into(),
                name: Some("shell".into()),
                content: "a.txt\nb.txt".into(),
                images: Vec::new(),
            },
        ];
        store.archive_window(1, &msgs).unwrap();

        let loaded = store.load_all_messages();
        assert_eq!(loaded.len(), 3);

        match &loaded[1] {
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(content.as_deref(), Some("looking"));
                let tcs = tool_calls
                    .as_ref()
                    .expect("tool_calls must survive the archive");
                assert_eq!(tcs.len(), 1);
                assert_eq!(tcs[0].id, "tc_1");
                assert_eq!(tcs[0].name, "shell");
                assert_eq!(tcs[0].arguments, "{\"cmd\":\"ls\"}");
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
        match &loaded[2] {
            ChatMessage::Tool {
                tool_call_id,
                name,
                content,
                images: _,
            } => {
                assert_eq!(tool_call_id, "tc_1");
                assert_eq!(name.as_deref(), Some("shell"));
                assert_eq!(content, "a.txt\nb.txt");
            }
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    #[test]
    fn load_all_messages_reads_legacy_flat_records() {
        // Archives written before the `msg` payload existed have only the flat
        // fields. Text must still come back; tool pairing is unrecoverable and
        // falls back to a synthetic call id (callers sanitize before resume).
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "legacy_session");
        let windows = tmp
            .path()
            .join("history")
            .join("legacy_session")
            .join("windows");
        fs::create_dir_all(&windows).unwrap();
        fs::write(
            windows.join("001.jsonl"),
            concat!(
                "{\"window\":1,\"index\":0,\"role\":\"user\",\"content\":\"old question\",\"ts\":\"t\"}\n",
                "{\"window\":1,\"index\":1,\"role\":\"assistant\",\"content\":\"old answer\",\"ts\":\"t\"}\n",
                "{\"window\":1,\"index\":2,\"role\":\"tool\",\"name\":\"shell\",\"content\":\"old output\",\"ts\":\"t\"}\n",
            ),
        )
        .unwrap();
        // metadata.json is what list_windows reads.
        fs::write(
            tmp.path().join("history").join("legacy_session").join("metadata.json"),
            "{\"session_id\":\"legacy_session\",\"windows\":[{\"id\":\"001\",\"started_at\":\"t\",\"message_count\":3}]}",
        )
        .unwrap();

        let loaded = store.load_all_messages();
        assert_eq!(loaded.len(), 3);
        assert!(
            matches!(&loaded[0], ChatMessage::User { content, .. } if content == "old question")
        );
        assert!(
            matches!(&loaded[1], ChatMessage::Assistant { content: Some(c), .. } if c == "old answer")
        );
        match &loaded[2] {
            ChatMessage::Tool {
                tool_call_id,
                name,
                content,
                images: _,
            } => {
                assert_eq!(tool_call_id, "archived-1-2");
                assert_eq!(name.as_deref(), Some("shell"));
                assert_eq!(content, "old output");
            }
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    #[test]
    fn load_all_messages_spans_windows_in_order() {
        let tmp = TempDir::new().unwrap();
        let store = HistoryStore::new(tmp.path(), "test_session");
        store
            .archive_window(2, &[ChatMessage::user("second window")])
            .unwrap();
        store
            .archive_window(1, &[ChatMessage::user("first window")])
            .unwrap();

        let loaded = store.load_all_messages();
        assert_eq!(loaded.len(), 2);
        assert!(
            matches!(&loaded[0], ChatMessage::User { content, .. } if content == "first window")
        );
        assert!(
            matches!(&loaded[1], ChatMessage::User { content, .. } if content == "second window")
        );
    }
}
