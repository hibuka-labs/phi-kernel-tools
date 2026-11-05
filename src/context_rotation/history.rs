//! History storage for context rotation.
//!
//! Archives context window messages to JSONL files and maintains
//! per-session metadata. Used by the rotation policy during
//! window resets.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use agent_base::ChatMessage;

use super::scope::SessionScope;

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
                // Legacy flat rows predate the synthetic flag and carry no
                // provenance — assume real (no guessing from content).
                synthetic: false,
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

/// Manages history storage at `<base_dir>/history/<session_id>/`.
///
/// The session tree is resolved at call time through a shared
/// [`SessionScope`], so one re-point (the `/resume` switch) moves every
/// subsequent write — a copied id froze the launch session and kept writing
/// its archive after a switch (observed 2026-11-04).
pub struct HistoryStore {
    scope: Arc<SessionScope>,
}

impl HistoryStore {
    /// Create a new store for the given base directory and session ID.
    ///
    /// `base_dir` is typically `~/.phimint/`. The session directory
    /// (`<base_dir>/history/<session_id>/`) is created on first write.
    /// The store is pinned to `session_id` for life — use
    /// [`HistoryStore::new_scoped`] to follow a re-point.
    pub fn new(base_dir: &Path, session_id: &str) -> Self {
        Self::new_scoped(Arc::new(SessionScope::new(base_dir, session_id)))
    }

    /// A store serving whatever session `scope` currently points at.
    pub fn new_scoped(scope: Arc<SessionScope>) -> Self {
        Self { scope }
    }

    /// The session's history tree — resolved at call time.
    fn session_dir(&self) -> PathBuf {
        self.scope.history_dir()
    }

    /// The highest archived window id on disk (0 when none).
    ///
    /// The JSONL filenames are the authority, not `metadata.json`: a crash
    /// between the two writes leaves the metadata stale, and the rotation
    /// counter must seed past what physically exists — an undercount
    /// truncates an existing `windows/NNN.jsonl` on the next archive.
    pub fn last_window_id(&self) -> usize {
        let windows_dir = self.session_dir().join("windows");
        let Ok(entries) = fs::read_dir(&windows_dir) else {
            return 0;
        };
        entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let stem = name.to_str()?.strip_suffix(".jsonl")?;
                stem.parse::<usize>().ok()
            })
            .max()
            .unwrap_or(0)
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
        let windows_dir = self.session_dir().join("windows");
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
        let mut degraded = 0usize;
        for (i, msg) in messages.iter().enumerate() {
            // Lossless copy: keeps `tool_calls` / `tool_call_id` so the
            // window can be replayed and restored, not just searched.
            // A serialization failure degrades to the flat fields rather
            // than losing the archive — but loudly: a degraded row can no
            // longer round-trip through [`MessageRecord::into_message`].
            let full = match serde_json::to_value(msg) {
                Ok(value) => Some(value),
                Err(e) => {
                    degraded += 1;
                    tracing::warn!(
                        error = %e,
                        window_id,
                        index = i,
                        "archived message lost its lossless payload; falling back to flat fields"
                    );
                    None
                }
            };
            let record = MessageRecord {
                window: window_id,
                index: i,
                role: message_role(msg),
                name: message_name(msg),
                content: message_content(msg),
                ts: ts.clone(),
                msg: full,
            };
            writeln!(file, "{}", serde_json::to_string(&record)?)?;
        }
        file.flush()?;
        if degraded > 0 {
            tracing::warn!(
                window_id,
                degraded,
                total = messages.len(),
                "archive window has degraded rows; replay/restore falls back to flat text"
            );
        }

        // Update metadata
        self.update_metadata(window_id, messages.len())?;

        Ok(messages.len())
    }

    /// Read the session metadata, or create a default if it doesn't exist.
    pub fn load_metadata(&self) -> SessionMeta {
        let meta_path = self.session_dir().join("metadata.json");
        let empty = SessionMeta {
            session_id: self.session_id(),
            windows: Vec::new(),
        };
        match fs::read_to_string(&meta_path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or(empty),
            Err(_) => empty,
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

        let session_dir = self.session_dir();
        let meta_path = session_dir.join("metadata.json");
        fs::create_dir_all(&session_dir)?;
        fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    }

    fn session_id(&self) -> String {
        self.scope.session_id()
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
        let path = self.session_dir().join("windows").join(&file_name);
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

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
