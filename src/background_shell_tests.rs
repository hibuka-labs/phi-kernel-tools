use super::*;

/// Helper: build a registry + cancel token.
fn setup(max_tasks: usize) -> (Arc<BackgroundTaskRegistry>, CancellationToken) {
    let reg = BackgroundTaskRegistry::new(max_tasks);
    let token = CancellationToken::new();
    (reg, token)
}

// ── register ──

#[test]
fn register_returns_bg_id() {
    let (reg, token) = setup(4);
    let id = reg.register("echo hi", None, token, None, 0).unwrap();
    assert!(id.starts_with("bg_"), "id should start with bg_: {id}");
    assert_eq!(id.len(), 11, "bg_ + 8 hex chars = 11");
}

// ── not_found teaching message ──

#[test]
fn not_found_message_redirects_sub_agent_polling() {
    // Session 20260919_f65b754c: 12 × `bg_analyze-*` polls for reports
    // that are pushed, not fetched. The error must name the redirect.
    let msg = not_found_message("bg_analyze-codex");
    assert!(msg.contains("bg_analyze-codex"), "id echoed: {msg}");
    assert!(msg.contains("not found"), "fact first: {msg}");
    assert!(msg.contains("sub-agent"), "names the confusion: {msg}");
    assert!(msg.contains("end your turn"), "names the action: {msg}");
}

// ── render_status self-truncation ──

fn done_snapshot(stdout: &str, stderr: &str) -> BackgroundTaskSnapshot {
    BackgroundTaskSnapshot {
        id: "bg_deadbeef".into(),
        command: "big".into(),
        working_dir: None,
        timeout_ms: 0,
        status: BackgroundTaskStatus::Done,
        started_at: Instant::now(),
        finished_at: Some(Instant::now()),
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
        exit_code: Some(0),
        consumed: false,
    }
}

/// A fat task result must self-truncate to the engine's output budget
/// (head+tail + marker) instead of dying as `ToolOutputTooLarge` (session
/// 20261018_7844215b).
#[test]
fn render_status_self_truncates_to_budget() {
    let snap = done_snapshot(&format!("HEAD{}", "x".repeat(40_000)), &"e".repeat(40_000));
    let text = render_status(&snap, Some(2_000), None);
    assert!(
        text.chars().count() <= 2_000,
        "over budget: {}",
        text.chars().count()
    );
    assert!(text.contains("output truncated"), "marker missing:\n{text}");
}

#[test]
fn render_status_unbounded_keeps_full_output() {
    let text = render_status(&done_snapshot("hello", "err"), None, None);
    assert!(text.contains("hello"), "stdout dropped:\n{text}");
    assert!(text.contains("err"), "stderr dropped:\n{text}");
}

#[test]
fn register_records_command_and_dir() {
    let (reg, token) = setup(4);
    let id = reg
        .register("ls -la", Some("/tmp"), token, None, 120_000)
        .unwrap();
    let snap = reg.snapshot(&id).unwrap();
    assert_eq!(snap.command, "ls -la");
    assert_eq!(snap.working_dir.as_deref(), Some("/tmp"));
    assert_eq!(snap.timeout_ms, 120_000, "timeout fuse must be recorded");
    assert_eq!(snap.status, BackgroundTaskStatus::Running);
}

#[test]
fn register_records_indefinite_daemon_as_zero_timeout() {
    // timeout_ms == 0 is the daemon marker ("start and leave alone") —
    // the UI classifies on it, so it must survive registration verbatim.
    let (reg, token) = setup(4);
    let id = reg
        .register("mvn spring-boot:run", None, token, None, 0)
        .unwrap();
    let snap = reg.snapshot(&id).unwrap();
    assert_eq!(snap.timeout_ms, 0, "0 = indefinite, never reinterpreted");
}

#[test]
fn register_respects_max_tasks() {
    let (reg, _token) = setup(2);
    let t1 = CancellationToken::new();
    let t2 = CancellationToken::new();
    let t3 = CancellationToken::new();
    reg.register("cmd1", None, t1, None, 0).unwrap();
    reg.register("cmd2", None, t2, None, 0).unwrap();
    let err = reg.register("cmd3", None, t3, None, 0).unwrap_err();
    assert!(err.contains("too many"));
}

#[test]
fn register_allows_new_after_terminal() {
    let (reg, _) = setup(2);
    let t1 = CancellationToken::new();
    let t2 = CancellationToken::new();
    let t3 = CancellationToken::new();
    let id1 = reg.register("cmd1", None, t1, None, 0).unwrap();
    reg.register("cmd2", None, t2, None, 0).unwrap();
    reg.finish(&id1, Some(0));
    reg.register("cmd3", None, t3, None, 0).unwrap();
}

// ── finish / update_status ──

#[test]
fn finish_sets_done_and_exit_code() {
    let (reg, token) = setup(4);
    let id = reg.register("cmd", None, token, None, 0).unwrap();
    reg.finish(&id, Some(42));
    let snap = reg.snapshot(&id).unwrap();
    assert_eq!(snap.status, BackgroundTaskStatus::Done);
    assert_eq!(snap.exit_code, Some(42));
    assert!(snap.finished_at.is_some());
}

#[test]
fn terminal_state_protection() {
    let (reg, token) = setup(4);
    let id = reg.register("cmd", None, token, None, 0).unwrap();
    reg.finish(&id, Some(0));
    // Try to update to Error — should be ignored.
    reg.set_error(&id, "boom".into());
    let snap = reg.snapshot(&id).unwrap();
    assert_eq!(snap.status, BackgroundTaskStatus::Done);
}

// ── snapshot_all / GC ──

#[test]
fn gc_sweeps_by_consume_then_fallback_ttl() {
    let (reg, token) = setup(4);
    let id = reg.register("cmd", None, token, None, 0).unwrap();
    reg.finish(&id, Some(0));

    // Pure read keeps it; long TTL keeps it; consumed → swept next gc().
    assert_eq!(reg.snapshot_all().len(), 1);
    reg.gc(Duration::from_secs(1800));
    assert_eq!(reg.snapshot_all().len(), 1);
    assert!(reg.mark_consumed(&id));
    reg.gc(Duration::from_secs(1800));
    assert_eq!(reg.snapshot_all().len(), 0);
}

// ── wake tail formatting ──

#[test]
fn tail_utf8_is_capped_and_never_splits_a_code_point() {
    // 4-byte emoji straddling the byte cut must not panic or mojibake.
    let s = "日本語テキスト🚀🚀🚀🚀";
    let t = tail_utf8(s, 7);
    assert!(t.len() <= 7);
    assert!(s.ends_with(t), "tail is a suffix");
    assert!(!t.is_empty());

    assert_eq!(tail_utf8("hello", 100), "hello");
    assert_eq!(tail_utf8("hello", 0), "");
    assert_eq!(tail_utf8("", 10), "");

    // Hard cap case: byte length exceeds the cap but result stays valid.
    let big = "あ".repeat(100);
    let t = tail_utf8(&big, 10);
    assert!(t.len() <= 10);
    assert!(t.len() >= 9, "clamped to the boundary right after the cut");
}

#[test]
fn output_tail_merges_stdout_and_stderr_with_cap() {
    let snap = BackgroundTaskSnapshot {
        id: "bg_x".into(),
        command: "cmd".into(),
        working_dir: None,
        timeout_ms: 0,
        status: BackgroundTaskStatus::Done,
        started_at: Instant::now(),
        finished_at: Some(Instant::now()),
        stdout: "line1\nline2\n".into(),
        stderr: "boom\n".into(),
        exit_code: Some(0),
        consumed: false,
    };
    let tail = snap.output_tail(1024);
    assert!(tail.contains("line1"), "stdout kept: {tail}");
    assert!(tail.contains("[stderr]"), "stderr section marked");
    assert!(tail.contains("boom"), "stderr kept");

    let capped = snap.output_tail(6);
    assert!(capped.len() <= 6, "cap enforced: {:?}", capped);
}

/// Multibyte-heavy payloads must not underflow (the old `char_count - TAIL_LIMIT`
/// underflowed usize when bytes >32K but chars <24K: debug panic / release
/// silently dropped the tail).
#[test]
fn bound_buffer_multibyte_heavy_does_not_underflow() {
    // 40 CJK chars per line = 120 bytes / 40 chars; 1000 lines = 120KB bytes / 40K chars.
    // A char count > 32K would also trigger bounding — so squeeze down to the
    // critical mix where chars stay < 24K: 200 lines = 24KB bytes (not over the
    // limit), use mixed line lengths so bytes cross the limit while chars don't.
    let line = "あ".repeat(40); // 40 chars, 120 bytes
    let mut buf = String::new();
    for _ in 0..300 {
        buf.push_str(&line);
        buf.push('\n');
    } // ≈36KB bytes, ≈12.3K chars: the old impl panicked / dropped the tail here
    let total_chars = buf.chars().count();
    assert!(buf.len() > 32 * 1024, "byte guard must fire");
    assert!(total_chars < 24 * 1024, "char count below old TAIL_LIMIT");

    ShellTaskEntry::bound_buffer(&mut buf);

    assert!(buf.len() < 40 * 1024, "bounded: {} bytes", buf.len());
    assert!(buf.contains("...[truncated]..."), "marker present");
    assert!(buf.ends_with(&(line.clone() + "\n")), "tail kept");
    assert!(buf.starts_with(&line), "head kept");
}

#[test]
fn bound_buffer_ascii_keeps_head_and_tail() {
    let line = "abcdefghij\n".to_string(); // 11 bytes
    let mut buf = line.repeat(5000); // 55KB
    ShellTaskEntry::bound_buffer(&mut buf);
    assert!(buf.len() <= 8 * 1024 + 24 * 1024 + 32, "bounded");
    assert!(buf.starts_with("abcdefghij"), "head kept");
    assert!(buf.contains("...[truncated]..."), "marker present");
    assert!(buf.ends_with("abcdefghij\n"), "tail kept");
}

// ── G3: task_output overflow spill ───────────────────────────────────

use crate::spill::StubSink;
use agent_base::tool::content_text;

/// protects: G3b on `task_output` — a status envelope over budget spills the
/// full pre-truncation status JSON (kind `shell_output`) and the result
/// marker names the artifact with the read_file continuation hint.
/// fails_when: the artifact is missing or holds trimmed fields, or the marker
/// carries no path.
/// no_claim: no-sink degradation (render cards) or execute_command (its card).
#[tokio::test]
async fn task_output_over_cap_spills_full_status_json() {
    let dir = tempfile::tempdir().unwrap();
    let sink = StubSink::new(dir.path());
    let (reg, token) = setup(4);
    let id = reg.register("big", None, token, None, 0).unwrap();
    reg.append_output(&id, Some(&"x".repeat(30_000)), Some(&"e".repeat(30_000)));
    reg.finish(&id, Some(0));

    let tool = TaskOutputTool::new(reg).with_spill(sink.clone());
    let mut ctx = ToolContext::for_test();
    ctx.max_output_chars = Some(2_000);

    let out = tool.call(&json!({"task_id": id}), &ctx).await.unwrap();
    let text = content_text(&out);
    assert!(
        text.contains("full output:"),
        "path marker missing:\n{text:.400}"
    );
    assert!(
        text.contains("(use read_file with offset/limit)"),
        "continuation hint missing:\n{text:.400}"
    );

    let calls = sink.calls();
    assert_eq!(calls.len(), 1, "exactly one spill per cut");
    assert_eq!(calls[0].kind, "shell_output");
    // The spill body is the full untrimmed status JSON: its payload fields
    // hold the whole buffered output, not the budget-cut fields.
    let v: Value = serde_json::from_str(&calls[0].body).expect("spill body is valid JSON");
    assert_eq!(v["status"], "done");
    assert!(
        v["stdout"].as_str().unwrap().chars().count() > 2_000,
        "spilled stdout must not be the truncated field"
    );
    assert!(
        v["stderr"].as_str().unwrap().chars().count() > 2_000,
        "spilled stderr must not be the truncated field"
    );
    // Fetch round-trip: artifact bytes == recorded body.
    assert_eq!(
        std::fs::read_to_string(&calls[0].path).unwrap(),
        calls[0].body
    );
}
