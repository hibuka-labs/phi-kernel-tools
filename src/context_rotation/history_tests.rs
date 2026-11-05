//! Tests for [`super`] — history storage (split out of `history.rs` to keep
//! every source file under the 1000-line rule; `#[path]` keeps private-item
//! access).

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
    assert!(matches!(&loaded[0], ChatMessage::User { content, .. } if content == "old question"));
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
    assert!(matches!(&loaded[0], ChatMessage::User { content, .. } if content == "first window"));
    assert!(matches!(&loaded[1], ChatMessage::User { content, .. } if content == "second window"));
}

// ── Lossless-archive assertion: the window must come back as it went in ──

#[test]
fn archive_round_trips_the_full_message_shapes() {
    // Every shape the engine can hold, archived and rebuilt: compared as
    // wire values so the never-persisted `ephemeral` flags don't count as
    // loss. A regression here means a window can no longer be restored.
    let tmp = TempDir::new().unwrap();
    let store = HistoryStore::new(tmp.path(), "test_session");
    let msgs = vec![
        ChatMessage::system("You are phimint, a coding agent."),
        ChatMessage::user("sow the nursing detail page"),
        ChatMessage::user_synthetic("Tool calls failed: exec_command"),
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
            content: "a.txt".into(),
            images: Vec::new(),
        },
        ChatMessage::Custom {
            role: "note".into(),
            data: serde_json::json!({ "content": "hi" }),
        },
    ];
    store.archive_window(1, &msgs).unwrap();

    let loaded = store.load_window_messages(1);
    assert_eq!(loaded.len(), msgs.len(), "one row in, one message out");
    for (i, (before, after)) in msgs.iter().zip(&loaded).enumerate() {
        assert_eq!(
            serde_json::to_value(before).unwrap(),
            serde_json::to_value(after).unwrap(),
            "message {i} changed across the archive"
        );
    }
    // Provenance specifically: a synthetic seed stays synthetic after
    // restore, so a re-rotated window keeps it off the user trail.
    assert!(matches!(
        &loaded[2],
        ChatMessage::User {
            synthetic: true,
            ..
        }
    ));
}

/// Regression fixtures copied from the head of session 20261103_4206f951's
/// `windows/002.jsonl` and `004.jsonl` — the resumed windows that archived
/// with a 3-row head. Rows written before the `synthetic` flag existed carry
/// no key and decode as whatever the user role says: provenance is never
/// guessed from content, even when the text is obviously a machine notice.
#[test]
fn real_window_head_rows_decode_with_provenance() {
    let tmp = TempDir::new().unwrap();
    let store = HistoryStore::new(tmp.path(), "20261103_4206f951");
    let windows = tmp
        .path()
        .join("history")
        .join("20261103_4206f951")
        .join("windows");
    fs::create_dir_all(&windows).unwrap();
    fs::write(
        windows.join("002.jsonl"),
        concat!(
            "{\"window\":2,\"index\":0,\"role\":\"system\",\"content\":\"You are phimint, a coding agent.\",\"ts\":\"t\",\"msg\":{\"System\":{\"content\":\"You are phimint, a coding agent.\"}}}\n",
            "{\"window\":2,\"index\":1,\"role\":\"user\",\"content\":\"@pigsight-cloud/ 你帮我看下，接口要加到这个工程\",\"ts\":\"t\",\"msg\":{\"User\":{\"content\":\"@pigsight-cloud/ 你帮我看下，接口要加到这个工程\"}}}\n",
            "{\"window\":2,\"index\":2,\"role\":\"user\",\"content\":\"Your previous context window was archived.\",\"ts\":\"t\",\"msg\":{\"User\":{\"content\":\"Your previous context window was archived.\"}}}\n",
        ),
    )
    .unwrap();
    fs::write(
        windows.join("004.jsonl"),
        concat!(
            // The W4 head: a machine "Tool calls failed" notice archived as
            // the last user message, untagged on disk.
            "{\"window\":4,\"index\":0,\"role\":\"system\",\"content\":\"You are phimint, a coding agent.\",\"ts\":\"t\",\"msg\":{\"System\":{\"content\":\"You are phimint, a coding agent.\"}}}\n",
            "{\"window\":4,\"index\":1,\"role\":\"user\",\"content\":\"Tool calls failed: exec_command\",\"ts\":\"t\",\"msg\":{\"User\":{\"content\":\"Tool calls failed: exec_command\"}}}\n",
            "{\"window\":4,\"index\":2,\"role\":\"user\",\"content\":\"Your previous context window was archived.\",\"ts\":\"t\",\"msg\":{\"User\":{\"content\":\"Your previous context window was archived.\"}}}\n",
        ),
    )
    .unwrap();

    let w2 = store.load_window_messages(2);
    assert_eq!(w2.len(), 3);
    assert!(
        matches!(&w2[0], ChatMessage::System { content, .. } if content == "You are phimint, a coding agent.")
    );
    assert!(matches!(
        &w2[1],
        ChatMessage::User {
            synthetic: false,
            ..
        }
    ));
    assert!(matches!(
        &w2[2],
        ChatMessage::User {
            synthetic: false,
            ..
        }
    ));

    let w4 = store.load_window_messages(4);
    assert!(
        matches!(&w4[1], ChatMessage::User { content, synthetic: false, .. }
            if content == "Tool calls failed: exec_command")
    );

    // A row that DOES carry the tag keeps it across the archive.
    fs::write(
        windows.join("005.jsonl"),
        "{\"window\":5,\"index\":0,\"role\":\"user\",\"content\":\"Tool calls failed: exec_command\",\"ts\":\"t\",\"msg\":{\"User\":{\"content\":\"Tool calls failed: exec_command\",\"synthetic\":true}}}\n",
    )
    .unwrap();
    let w5 = store.load_window_messages(5);
    assert!(matches!(
        &w5[0],
        ChatMessage::User {
            synthetic: true,
            ..
        }
    ));
}

// ── SessionScope follow: the store must track a `/resume` re-point ──

/// Regression (2026-11-04): a store frozen on the launch session kept
/// writing one session's archive into another's directory after a mid-session
/// `/resume`. A scoped store resolves its tree at call time, so a re-point
/// moves every subsequent write.
#[test]
fn scoped_store_follows_repoint() {
    let tmp = TempDir::new().unwrap();
    let scope = Arc::new(SessionScope::new(tmp.path(), "bootstrap"));
    let store = HistoryStore::new_scoped(Arc::clone(&scope));

    // Work in the launch session...
    store.archive_window(1, &make_messages(2)).unwrap();
    assert!(
        tmp.path()
            .join("history/bootstrap/windows/001.jsonl")
            .exists()
    );

    // ...then switch: the same store must now write the resumed session's
    // tree, and keep its hands off the abandoned one.
    scope.repoint("resumed");
    store.archive_window(1, &make_messages(3)).unwrap();
    assert!(
        tmp.path()
            .join("history/resumed/windows/001.jsonl")
            .exists()
    );
    assert_eq!(
        fs::read_to_string(tmp.path().join("history/bootstrap/windows/001.jsonl"))
            .unwrap()
            .lines()
            .count(),
        2,
        "the abandoned session's archive must not be touched"
    );
}

/// `last_window_id` is the rotation counter's seed: it must read what is
/// physically on disk (the JSONL filenames), 0 when the tree is empty.
#[test]
fn last_window_id_reads_the_archive() {
    let tmp = TempDir::new().unwrap();
    let store = HistoryStore::new(tmp.path(), "s");
    assert_eq!(store.last_window_id(), 0, "empty tree seeds at window 1");
    store.archive_window(1, &make_messages(1)).unwrap();
    store.archive_window(2, &make_messages(1)).unwrap();
    assert_eq!(store.last_window_id(), 2);
    // Filenames are the authority: a stale metadata.json must not undercount.
    fs::write(
        tmp.path().join("history/s/metadata.json"),
        "{\"session_id\":\"s\",\"windows\":[]}",
    )
    .unwrap();
    assert_eq!(store.last_window_id(), 2);
}
