//! Tests for [`super`] — edit_file helpers and tool integration (split out of
//! `edit_file.rs` to keep every source file under the 1000-line rule; `#[path]`
//! keeps private-item access).

use super::*;
use agent_base::tool::content_text;

fn dummy_ctx() -> ToolContext {
    ToolContext::for_test()
}

fn setup_temp_workspace() -> (tempfile::TempDir, EditFileTool) {
    let dir = tempfile::tempdir().unwrap();
    let tool = EditFileTool::new(dir.path().to_path_buf());
    (dir, tool)
}

// ── find_all_positions ──

#[test]
fn test_find_all_positions_single() {
    let pos = find_all_positions("hello world", "world");
    assert_eq!(pos, vec![6]);
}

#[test]
fn test_find_all_positions_multiple() {
    let pos = find_all_positions("foo bar foo baz foo", "foo");
    assert_eq!(pos, vec![0, 8, 16]);
}

#[test]
fn test_find_all_positions_multibyte_needle_no_panic() {
    // Bug-1: multi-byte UTF-8 needle (é = 2 bytes) causes start to land
    // mid-codepoint on the next iteration, panicking at "char boundary".
    // This test MUST pass after the fix.
    let pos = find_all_positions("aééb", "é");
    // "aééb" bytes: [0x61, 0xC3, 0xA9, 0xC3, 0xA9, 0x62]
    //                       ^1       ^3
    assert_eq!(pos, vec![1, 3]);
}

#[test]
fn test_find_all_positions_emoji_needle() {
    // Emoji: 🔥 = 4 bytes (0xF0 0x9F 0x94 0xA5)
    let pos = find_all_positions("x🔥y🔥z", "🔥");
    assert_eq!(pos, vec![1, 6]);
}

#[test]
fn test_find_all_positions_none() {
    let pos = find_all_positions("hello", "xyz");
    assert!(pos.is_empty());
}

#[test]
fn test_find_all_positions_empty_needle() {
    let pos = find_all_positions("hello", "");
    assert!(pos.is_empty());
}

// ── detect_line_ending ──

#[test]
fn test_detect_line_ending_lf() {
    assert_eq!(detect_line_ending("hello\nworld\n"), "\n");
}

#[test]
fn test_detect_line_ending_crlf() {
    assert_eq!(detect_line_ending("hello\r\nworld\r\n"), "\r\n");
}

#[test]
fn test_detect_line_ending_mixed_prefers_crlf() {
    assert_eq!(detect_line_ending("hello\nworld\r\n"), "\r\n");
}

// ── normalize_line_endings ──

#[test]
fn test_normalize_to_lf() {
    assert_eq!(
        normalize_line_endings("hello\r\nworld\r\n", "\n"),
        "hello\nworld\n"
    );
}

#[test]
fn test_normalize_to_crlf() {
    assert_eq!(
        normalize_line_endings("hello\nworld\n", "\r\n"),
        "hello\r\nworld\r\n"
    );
}

#[test]
fn test_normalize_no_change() {
    assert_eq!(
        normalize_line_endings("hello\nworld\n", "\n"),
        "hello\nworld\n"
    );
}

// ── apply_replace ──

#[test]
fn test_apply_replace_basic() {
    let result = apply_replace("hello world", 6, "world", "earth");
    assert_eq!(result, "hello earth");
}

#[test]
fn test_apply_replace_beginning() {
    let result = apply_replace("fn old() {}", 3, "old", "new");
    assert_eq!(result, "fn new() {}");
}

// ── check_overlaps ──

#[test]
fn test_check_overlaps_no_overlap() {
    let content = "line1\nline2\nline3\n";
    let edits = vec![
        Edit {
            old_text: "line1".to_string(),
            new_text: "LINE1".to_string(),
        },
        Edit {
            old_text: "line3".to_string(),
            new_text: "LINE3".to_string(),
        },
    ];
    assert!(check_overlaps(content, &edits).is_ok());
}

#[test]
fn test_check_overlaps_detected() {
    let content = "hello world";
    let edits = vec![
        Edit {
            old_text: "hello world".to_string(),
            new_text: "hi earth".to_string(),
        },
        Edit {
            old_text: "world".to_string(),
            new_text: "earth".to_string(),
        },
    ];
    let err = check_overlaps(content, &edits).unwrap_err();
    assert!(err.contains("overlap"));
}

// ── find_and_replace ──

#[test]
fn test_find_and_replace_exact() {
    let result = find_and_replace(
        "hello world",
        "hello world",
        &Edit {
            old_text: "world".to_string(),
            new_text: "earth".to_string(),
        },
        0,
    )
    .unwrap();
    assert_eq!(result, "hello earth");
}

#[test]
fn test_find_and_replace_not_unique() {
    let err = find_and_replace(
        "foo bar foo",
        "foo bar foo",
        &Edit {
            old_text: "foo".to_string(),
            new_text: "baz".to_string(),
        },
        0,
    )
    .unwrap_err();
    assert!(err.contains("not unique"));
}

#[test]
fn test_find_and_replace_not_found() {
    let err = find_and_replace(
        "hello world",
        "hello world",
        &Edit {
            old_text: "xyz".to_string(),
            new_text: "abc".to_string(),
        },
        0,
    )
    .unwrap_err();
    assert!(err.contains("not found"));
}

#[test]
fn test_find_and_replace_ambiguity_after_prior_edits() {
    // current has two "foo" (from a prior edit), original had only one.
    let err = find_and_replace(
        "foo foo",
        "foo bar",
        &Edit {
            old_text: "foo".to_string(),
            new_text: "baz".to_string(),
        },
        0,
    )
    .unwrap_err();
    assert!(err.contains("after previous edits"));
}

#[test]
fn test_find_and_replace_rstrip_single() {
    // old_text has trailing whitespace not present exactly in current,
    // but the rstripped form matches once.
    let result = find_and_replace(
        "hello world  x",
        "hello world  x",
        &Edit {
            old_text: "world \n".to_string(),
            new_text: "earth".to_string(),
        },
        0,
    )
    .unwrap();
    assert_eq!(result, "hello earthx");
}

#[test]
fn test_find_and_replace_rstrip_multiple() {
    let err = find_and_replace(
        "world  x world  y",
        "world  x world  y",
        &Edit {
            old_text: "world \n".to_string(),
            new_text: "earth".to_string(),
        },
        0,
    )
    .unwrap_err();
    assert!(err.contains("trailing whitespace stripped"));
}

#[test]
fn test_find_and_replace_trim_single() {
    // old_text has leading+trailing whitespace; trimmed form matches once.
    let result = find_and_replace(
        "hello  world  x",
        "hello  world  x",
        &Edit {
            old_text: "   world  ".to_string(),
            new_text: "earth".to_string(),
        },
        0,
    )
    .unwrap();
    assert_eq!(result, "helloearthx");
}

#[test]
fn test_find_and_replace_trim_multiple() {
    let err = find_and_replace(
        "  world  x  world  y",
        "  world  x  world  y",
        &Edit {
            old_text: "   world  ".to_string(),
            new_text: "earth".to_string(),
        },
        0,
    )
    .unwrap_err();
    assert!(err.contains("whitespace trimmed"));
}

#[test]
fn test_find_nfc_matches() {
    // unicode_normalize is currently identity, so NFC matching degenerates
    // to a plain substring match — still exercised to lock in behaviour.
    let m = find_nfc_matches("héllo wörld", "wörld");
    assert_eq!(m.len(), 1);
    assert_eq!(m[0].1, "wörld");
}

// ── Full tool integration tests ──

#[tokio::test]
async fn test_edit_file_single_edit() {
    let (dir, tool) = setup_temp_workspace();
    std::fs::write(dir.path().join("test.txt"), "hello world\n").unwrap();

    let result = tool
        .call(
            &json!({
                "path": "test.txt",
                "edits": [{"old_text": "hello", "new_text": "hi"}]
            }),
            &dummy_ctx(),
        )
        .await
        .unwrap();

    assert!(content_text(&result).contains("Successfully applied 1 edit"));
    let content = std::fs::read_to_string(dir.path().join("test.txt")).unwrap();
    assert_eq!(content, "hi world\n");
}

#[tokio::test]
async fn test_edit_file_multiple_edits() {
    let (dir, tool) = setup_temp_workspace();
    std::fs::write(dir.path().join("test.rs"), "fn old() {}\nfn other() {}\n").unwrap();

    let result = tool
        .call(
            &json!({
                "path": "test.rs",
                "edits": [
                    {"old_text": "fn old() {}", "new_text": "fn new() {}"},
                    {"old_text": "fn other() {}", "new_text": "fn another() {}"}
                ]
            }),
            &dummy_ctx(),
        )
        .await
        .unwrap();

    assert!(content_text(&result).contains("Successfully applied 2 edit"));
    let content = std::fs::read_to_string(dir.path().join("test.rs")).unwrap();
    assert_eq!(content, "fn new() {}\nfn another() {}\n");
}

#[tokio::test]
async fn test_edit_file_not_found() {
    let (_dir, tool) = setup_temp_workspace();
    let result = tool
        .call(
            &json!({
                "path": "nonexistent.txt",
                "edits": [{"old_text": "x", "new_text": "y"}]
            }),
            &dummy_ctx(),
        )
        .await
        .unwrap();
    assert!(content_text(&result).contains("not found"));
}

#[tokio::test]
async fn test_edit_file_duplicate_old_text() {
    let (dir, tool) = setup_temp_workspace();
    std::fs::write(dir.path().join("dup.txt"), "foo bar foo\n").unwrap();

    let result = tool
        .call(
            &json!({
                "path": "dup.txt",
                "edits": [{"old_text": "foo", "new_text": "baz"}]
            }),
            &dummy_ctx(),
        )
        .await
        .unwrap();

    assert!(content_text(&result).contains("not unique"));
}

#[tokio::test]
async fn test_edit_file_no_path() {
    let (_dir, tool) = setup_temp_workspace();
    let result = tool
        .call(
            &json!({"edits": [{"old_text": "x", "new_text": "y"}]}),
            &dummy_ctx(),
        )
        .await
        .unwrap();
    assert!(content_text(&result).contains("No file path"));
}

#[tokio::test]
async fn test_edit_file_no_edits() {
    let (_dir, tool) = setup_temp_workspace();
    let result = tool
        .call(&json!({"path": "test.txt"}), &dummy_ctx())
        .await
        .unwrap();
    assert!(content_text(&result).contains("No edits provided"));
}

#[tokio::test]
async fn test_edit_file_empty_edits() {
    let (_dir, tool) = setup_temp_workspace();
    let result = tool
        .call(&json!({"path": "test.txt", "edits": []}), &dummy_ctx())
        .await
        .unwrap();
    assert!(content_text(&result).contains("empty"));
}

#[tokio::test]
async fn test_edit_file_preserves_line_endings_crlf() {
    let (dir, tool) = setup_temp_workspace();
    std::fs::write(dir.path().join("crlf.txt"), "line1\r\nline2\r\nline3\r\n").unwrap();

    let result = tool
        .call(
            &json!({
                "path": "crlf.txt",
                "edits": [{"old_text": "line2", "new_text": "LINE2"}]
            }),
            &dummy_ctx(),
        )
        .await
        .unwrap();

    assert!(content_text(&result).contains("Successfully applied 1 edit"));
    let content = std::fs::read_to_string(dir.path().join("crlf.txt")).unwrap();
    assert_eq!(content, "line1\r\nLINE2\r\nline3\r\n");
}

#[tokio::test]
async fn test_edit_file_overlapping_edits_rejected() {
    let (dir, tool) = setup_temp_workspace();
    std::fs::write(dir.path().join("overlap.txt"), "hello world\n").unwrap();

    let result = tool
        .call(
            &json!({
                "path": "overlap.txt",
                "edits": [
                    {"old_text": "hello world", "new_text": "hi"},
                    {"old_text": "world", "new_text": "earth"}
                ]
            }),
            &dummy_ctx(),
        )
        .await
        .unwrap();

    assert!(content_text(&result).contains("overlap"));
}

#[tokio::test]
async fn test_edit_file_absolute_path_outside_workspace() {
    let (_dir, tool) = setup_temp_workspace();
    // No sandbox: an absolute path outside the workspace is editable.
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("outside.txt");
    std::fs::write(&target, "hello world\n").unwrap();

    let result = tool
        .call(
            &json!({
                "path": target.to_str().unwrap(),
                "edits": [{"old_text": "hello", "new_text": "hi"}]
            }),
            &dummy_ctx(),
        )
        .await
        .unwrap();

    assert!(content_text(&result).contains("Successfully applied 1 edit"));
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "hi world\n");
}

#[tokio::test]
async fn test_name_and_definition() {
    let tool = EditFileTool::new(PathBuf::from("/tmp"));
    assert_eq!(tool.name(), "edit_file");
    assert!(tool.description().contains("text replacements"));
    let schema = tool.schema();
    assert_eq!(schema["type"], "object");
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("path")));
    assert!(required.contains(&json!("edits")));
}

#[tokio::test]
async fn test_metadata() {
    let tool = EditFileTool::new(PathBuf::from("/tmp"));
    let meta = tool.metadata();
    assert_eq!(meta.name, "edit_file");
    assert_eq!(meta.origin, "phi-kernel-tools");
}

#[test]
fn test_find_rstrip_matches() {
    let content = "hello   \nworld\n";
    let matches = find_rstrip_matches(content, "hello");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].1, "hello   ");
}

#[test]
fn test_find_trim_matches() {
    let content = "  hello  \nworld\n";
    let matches = find_trim_matches(content, "hello");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].1, "  hello  ");
}

#[test]
fn test_level2_rstrip_fallback() {
    let content = "hello   \nworld\n";
    let result = find_and_replace(
        content,
        content,
        &Edit {
            old_text: "hello   ".to_string(),
            new_text: "hi".to_string(),
        },
        0,
    )
    .unwrap();
    assert_eq!(result, "hi\nworld\n");
}
