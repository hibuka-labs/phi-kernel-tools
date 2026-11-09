//! Tests for [`super`] — local shell tool (split out of `local_shell.rs` to
//! keep every source file under the 1000-line rule; `#[path]` keeps
//! private-item access).

use super::*;
use crate::task_registry::TaskStatus;
use agent_base::tool::content_text;

/// protects: G7 exit-code trailer on the stdout-only success arm.
/// fails_when: the trailer is missing, not the last line, or the body changed.
/// no_claim: failure-arm wording (see the vocabulary card).
#[test]
fn test_format_result_success() {
    let result = format_result("echo hello", "hello", "", Some(0), false);
    assert_eq!(result, "hello\n[exit code: 0]");
}

/// protects: G7 exit-code trailer on the stderr-only success arm.
/// fails_when: the trailer is missing, not the last line, or the body changed.
/// no_claim: failure-arm wording (see the vocabulary card).
#[test]
fn test_format_result_stderr_only() {
    let result = format_result("cmd", "", "error output", Some(0), false);
    assert_eq!(result, "stderr:\nerror output\n[exit code: 0]");
}

/// protects: G7 exit-code trailer on the both-streams success arm.
/// fails_when: the trailer is missing, not the last line, or the body changed.
/// no_claim: failure-arm wording (see the vocabulary card).
#[test]
fn test_format_result_stdout_and_stderr() {
    let result = format_result("cmd", "out", "err", Some(0), false);
    assert_eq!(result, "stdout:\nout\n\nstderr:\nerr\n[exit code: 0]");
}

/// protects: the terminated vocabulary is byte-stable under G7.
/// fails_when: any word of the `[Command Terminated]` envelope changes.
/// no_claim: success-arm behavior.
#[test]
fn test_format_result_terminated() {
    let result = format_result("cmd", "", "sigterm", None, false);
    assert_eq!(
        result,
        "[Command Terminated]\ncommand: cmd\nstdout:\n(empty)\nstderr:\nsigterm"
    );
}

/// protects: the empty-success sentence is byte-stable under G7 — no trailer
/// is appended when there is no output.
/// fails_when: the sentence gains a suffix, a period change, or a trailer.
/// no_claim: success-with-output arms.
#[test]
fn test_format_result_no_output() {
    let result = format_result("true", "", "", Some(0), false);
    assert_eq!(result, "Command executed successfully with no output.");
}

/// protects: the failure vocabulary is byte-stable under G7.
/// fails_when: the `[Command Failed (exit code: N)]` envelope changes shape.
/// no_claim: which exit codes map to which arms.
#[test]
fn test_format_result_failure() {
    let result = format_result("false", "", "error", Some(1), false);
    assert_eq!(
        result,
        "[Command Failed (exit code: 1)]\ncommand: false\nstdout:\n(empty)\nstderr:\nerror"
    );
}

/// protects: the timeout vocabulary is byte-stable under G7.
/// fails_when: the `[Command Timed Out]` envelope changes shape.
/// no_claim: timeout detection logic.
#[test]
fn test_format_result_timeout() {
    let result = format_result("sleep 100", "", "", None, true);
    assert_eq!(
        result,
        "[Command Timed Out]\ncommand: sleep 100\nstdout:\n(empty)\nstderr:\n(empty)"
    );
}

#[test]
fn test_name() {
    let tool = LocalShellTool::new(30000);
    assert_eq!(tool.name(), "execute_command");
}

#[test]
fn test_definition() {
    let tool = LocalShellTool::new(30000);
    assert_eq!(tool.name(), "execute_command");
    assert!(tool.description().contains("shell"));
    let schema = tool.schema();
    assert_eq!(schema["type"], "object");
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("command")));
}

#[test]
fn test_metadata() {
    let tool = LocalShellTool::new(30000);
    let meta = tool.metadata();
    assert_eq!(meta.name, "execute_command");
    assert_eq!(meta.origin, "phi-kernel-tools");
    assert_eq!(meta.version, env!("CARGO_PKG_VERSION"));
    assert!(meta.description.contains("shell"));
    assert!(meta.requirements.is_empty());
}

#[tokio::test]
async fn test_call_echo() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(&json!({"command": "echo hello"}), &ToolContext::for_test())
        .await
        .unwrap();
    assert!(content_text(&result).contains("hello"));
}

#[tokio::test]
async fn test_call_empty_command() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(&json!({}), &ToolContext::for_test())
        .await
        .unwrap();
    assert!(content_text(&result).contains("No command provided"));
}

#[tokio::test]
async fn test_call_failing_command() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(&json!({"command": "exit 3"}), &ToolContext::for_test())
        .await
        .unwrap();
    assert!(content_text(&result).contains("exit code: 3"));
}

#[tokio::test]
async fn test_call_working_dir() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(
            &json!({"command": "pwd", "working_dir": "/"}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(!text.contains("[Error]"));
    assert!(text.contains('/'));
}

#[tokio::test]
async fn test_call_timeout() {
    let tool = LocalShellTool::new(50);
    let result = tool
        .call(&json!({"command": "sleep 30"}), &ToolContext::for_test())
        .await
        .unwrap();
    assert!(content_text(&result).contains("Timed Out"));
}

/// A timeout must kill the whole process tree, not just the direct `sh`
/// child — otherwise a grandchild (e.g. a `java` server launched by `mvn`)
/// leaks as an orphan. Here `sleep` is the grandchild: `sh` backgrounds it
/// and `wait`s, so on timeout both must die.
#[tokio::test]
async fn test_call_timeout_kills_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("child.pid");
    let pidpath = pidfile.to_str().unwrap().to_string();

    // Background a long-lived `sleep` (grandchild), record its pid, then
    // `wait` so the direct child stays alive alongside it.
    let command = format!("sleep 60 & echo $! > '{pidpath}'; wait");
    let tool = LocalShellTool::new(300);
    let result = tool
        .call(&json!({"command": command}), &ToolContext::for_test())
        .await
        .unwrap();
    assert!(content_text(&result).contains("Timed Out"));

    // The grandchild's pid was recorded before the timeout; after the group
    // kill it must be gone. Poll — SIGKILL is async to the reporter, and a
    // loaded CI runner can delay the death observation, so allow a generous
    // 5s window (250 × 20ms) instead of a tight 1s poll.
    let pid: u32 = std::fs::read_to_string(&pidfile)
        .expect("grandchild pidfile should be written")
        .trim()
        .parse()
        .expect("pidfile should contain a pid");

    let mut alive = true;
    for _ in 0..250 {
        alive = std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !alive {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!alive, "grandchild pid {pid} survived the timeout kill");
}

/// Per-call `timeout_ms` must override the tool-level default (foreground).
#[tokio::test]
async fn test_call_timeout_override() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(
            &json!({"command": "sleep 30", "timeout_ms": 100}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    assert!(content_text(&result).contains("Timed Out"));
}

/// The pipeline watchdog budget must cover the per-call `timeout_ms` —
/// session 20261018_7844215b: a 660s promise was killed at the 600s
/// framework default because the watchdog never saw the argument. Slack
/// keeps the tool's own fuse the first to fire.
#[test]
fn call_timeout_ms_mirrors_per_call_budget() {
    let tool = LocalShellTool::new(120_000);
    assert_eq!(
        tool.call_timeout_ms(&json!({"command": "x", "timeout_ms": 660_000})),
        Some(660_000 + WATCHDOG_SLACK_MS)
    );
    assert_eq!(
        tool.call_timeout_ms(&json!({"command": "x", "timeout_ms": 0})),
        Some(0)
    );
    assert_eq!(
        tool.call_timeout_ms(&json!({"command": "x", "background": true})),
        Some(0)
    );
    assert_eq!(
        tool.call_timeout_ms(&json!({"command": "x"})),
        Some(120_000 + WATCHDOG_SLACK_MS)
    );
}

/// `sleep` as the *first* word is the turn-blocking wait anti-pattern and
/// earns a hint; `sleep` inside a real script stays untouched.
#[test]
fn sleep_poll_detection_is_hint_only() {
    assert!(is_sleep_poll("sleep 300; tail -4 log"));
    assert!(is_sleep_poll("  sleep 5"));
    assert!(!is_sleep_poll("./release.sh && sleep 1"));
    assert!(!is_sleep_poll("cargo test"));
}

#[tokio::test]
async fn sleep_poll_timeout_result_carries_hint() {
    let tool = LocalShellTool::new(50);
    let result = tool
        .call(&json!({"command": "sleep 30"}), &ToolContext::for_test())
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(text.contains("Timed Out"), "timeout not reported:\n{text}");
    assert!(text.contains("[hint]"), "hint missing:\n{text}");
}

/// `timeout_ms: 0` means "never kill": a command outliving the tool default
/// must finish instead of dying. Guards the from_millis(0) instant-kill
/// hazard (0 previously would have meant a 0ms fuse).
#[tokio::test]
async fn test_call_zero_timeout_is_unlimited() {
    let tool = LocalShellTool::new(50);
    let result = tool
        .call(
            &json!({"command": "sleep 0.5 && echo done", "timeout_ms": 0}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(
        !text.contains("Timed Out"),
        "0 must disable the fuse: {text}"
    );
    assert!(text.contains("done"));
}

/// Explicit `timeout_ms` must kill background tasks too — regression test
/// for the old code, which ignored the per-call timeout and fused every
/// background task at the tool-level default.
#[tokio::test]
async fn test_background_timeout_override() {
    let registry = BackgroundTaskRegistry::new(8);
    let tool = LocalShellTool::new(30000).with_registry(registry.clone());
    let result = tool
        .call(
            &json!({"command": "sleep 30", "background": true, "timeout_ms": 200}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let v: Value =
        serde_json::from_str(content_text(&result).trim()).expect("background call returns JSON");
    let tid = v["task_id"].as_str().unwrap().to_string();

    // Poll until terminal — 200ms fuse + kill/drain latency.
    let mut status = None;
    for _ in 0..150 {
        if let Some(snap) = registry.snapshot(&tid) {
            status = Some(snap.status.clone());
            if status.as_ref().unwrap().is_terminal() {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(status, Some(BackgroundTaskStatus::TimedOut));
}

/// Background default (no timeout_ms) = unlimited: the task must still be
/// Running past the tool-level fuse. This is what keeps dev servers alive
/// under background=true instead of dying at the hardcoded 120s.
#[tokio::test]
async fn test_background_default_runs_indefinitely() {
    let registry = BackgroundTaskRegistry::new(8);
    let tool = LocalShellTool::new(300).with_registry(registry.clone());
    let result = tool
        .call(
            &json!({"command": "sleep 30", "background": true}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let v: Value = serde_json::from_str(content_text(&result).trim()).unwrap();
    let tid = v["task_id"].as_str().unwrap().to_string();

    // Old behavior (fuse = tool default 300ms) would have killed it here.
    tokio::time::sleep(Duration::from_millis(600)).await;
    let snap = registry.snapshot(&tid).expect("task registered");
    assert_eq!(snap.status, BackgroundTaskStatus::Running);

    registry.cancel(&tid); // don't leak the sleeper
}

#[test]
fn test_truncate_output_unchanged_when_under_budget() {
    assert_eq!(truncate_output("short", Some(100)), "short");
    assert_eq!(truncate_output("short", None), "short");
}

#[test]
fn test_truncate_output_keeps_head_and_tail() {
    let s: String = (0..1000).map(|i| format!("L{i:03}\n")).collect();
    let out = truncate_output(&s, Some(200));
    let n = out.chars().count();
    assert!(n <= 200, "output exceeds budget: {n}");
    assert!(out.starts_with("L000"), "head missing:\n{out}");
    assert!(out.trim_end().ends_with("L999"), "tail missing:\n{out}");
    assert!(out.contains("output truncated"), "marker missing:\n{out}");
}

#[tokio::test]
async fn test_call_self_truncates_large_output() {
    let tool = LocalShellTool::new(30000);
    let mut ctx = ToolContext::for_test();
    ctx.max_output_chars = Some(200);
    let result = tool
        .call(&json!({"command": "seq 1 1000"}), &ctx)
        .await
        .unwrap();
    let text = content_text(&result);
    let n = text.chars().count();
    assert!(n <= 200, "output exceeds budget: {n}");
    assert!(text.contains("output truncated"), "marker missing:\n{text}");
    assert!(text.starts_with('1'), "head missing:\n{text}");
    // G7 shifts the summary's last line: the exit-code trailer is the tail of
    // the composed summary, the command's own tail ("1000") sits just above.
    assert!(
        text.trim_end().ends_with("[exit code: 0]"),
        "G7 trailer is not last:\n{text}"
    );
    assert!(text.contains("1000"), "command tail missing:\n{text}");
}

// ── G3: shell overflow spill ──────────────────────────────────────────

use crate::spill::StubSink;

/// protects: no spill below the cap — a summary that fits must never touch
/// the sink or grow a path marker.
/// fails_when: the sink sees a call, or the output differs from the input.
/// no_claim: over-cap behavior (see the cut cards).
#[test]
fn spill_under_cap_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let sink = StubSink::new(dir.path());
    let out = truncate_output_spilling("short", Some(100), Some(sink.as_ref()));
    assert_eq!(out, "short");
    assert!(sink.calls().is_empty(), "sink touched below the cap");
}

/// protects: G3b round-trip — a cut spills the full pre-truncation summary
/// byte-exact (kind `shell_output`) and the marker names the artifact path
/// with the read_file continuation hint.
/// fails_when: the spilled body differs by a byte, the marker lacks the path
/// or hint, or the marker is the plain one.
/// no_claim: sink-failure or no-sink paths.
#[test]
fn spill_cut_spills_full_body_and_marks_path() {
    let dir = tempfile::tempdir().unwrap();
    let sink = StubSink::new(dir.path());
    let full: String = (0..1000).map(|i| format!("L{i:03}\n")).collect();

    let out = truncate_output_spilling(&full, Some(200), Some(sink.as_ref()));

    assert!(out.contains("full output:"), "path marker missing:\n{out}");
    assert!(
        out.contains("(use read_file with offset/limit)"),
        "continuation hint missing:\n{out}"
    );
    let calls = sink.calls();
    assert_eq!(calls.len(), 1, "exactly one spill per cut");
    assert_eq!(calls[0].kind, "shell_output");
    // Fetch round-trip: the artifact is the pre-truncation body, byte-exact.
    let fetched = std::fs::read_to_string(&calls[0].path).unwrap();
    assert_eq!(fetched, full, "spilled body must be byte-identical");
    // The marker's path is the artifact the sink actually wrote.
    assert!(
        out.contains(&calls[0].path.display().to_string()),
        "marker path != artifact path:\n{out}"
    );
}

/// protects: G3b degradation — a sink write failure keeps today's plain
/// marker exactly. No panic, no fake path, no swallowed cut.
/// fails_when: the output differs from the no-sink truncation, or a path
/// appears without a real artifact.
/// no_claim: the sink's internal error taxonomy.
#[test]
fn spill_sink_failure_degrades_to_plain_marker() {
    let dir = tempfile::tempdir().unwrap();
    let sink = StubSink::new_failing(dir.path());
    let full: String = (0..1000).map(|i| format!("L{i:03}\n")).collect();

    let out = truncate_output_spilling(&full, Some(200), Some(sink.as_ref()));

    assert_eq!(
        out,
        truncate_output(&full, Some(200)),
        "sink failure must reproduce the no-sink output byte-for-byte"
    );
    assert!(!out.contains("full output:"), "fake path leaked:\n{out}");
}

/// protects: G3b no-sink identity — without an injected sink the truncation
/// output is byte-identical to today's.
/// fails_when: the marker or the head/tail split changes.
/// no_claim: with-sink behavior.
#[test]
fn spill_no_sink_is_plain_marker() {
    let full: String = (0..1000).map(|i| format!("L{i:03}\n")).collect();
    let out = truncate_output_spilling(&full, Some(200), None);
    assert_eq!(out, truncate_output(&full, Some(200)));
    assert!(out.contains(MARKER), "plain marker missing:\n{out}");
}

/// protects: G3b at the tool seam — `execute_command` over budget spills the
/// full composed summary (G7 trailer included) and the result marker carries
/// the artifact path + read_file hint.
/// fails_when: the artifact is missing, its body differs from the composed
/// summary, or the result marker has no path.
/// no_claim: task_output (its own card) or sink-failure (unit card).
#[tokio::test]
async fn call_over_cap_spills_full_summary_with_path_marker() {
    let dir = tempfile::tempdir().unwrap();
    let sink = StubSink::new(dir.path());
    let tool = LocalShellTool::new(30000).with_spill(sink.clone());
    let mut ctx = ToolContext::for_test();
    ctx.max_output_chars = Some(200);

    let result = tool
        .call(&json!({"command": "seq 1 1000"}), &ctx)
        .await
        .unwrap();
    let text = content_text(&result);

    assert!(
        text.contains("full output:"),
        "path marker missing:\n{text}"
    );
    assert!(
        text.contains("(use read_file with offset/limit)"),
        "continuation hint missing:\n{text}"
    );
    assert!(!text.contains(MARKER), "plain marker mixed in:\n{text}");

    let calls = sink.calls();
    assert_eq!(calls.len(), 1, "exactly one spill per cut");
    assert_eq!(calls[0].kind, "shell_output");
    // The spilled body is the composed pre-truncation summary: raw output
    // plus the G7 exit-code trailer, retrievable end to end.
    let fetched = std::fs::read_to_string(&calls[0].path).unwrap();
    assert_eq!(
        fetched, calls[0].body,
        "artifact body drifted from the record"
    );
    assert!(
        fetched.ends_with("[exit code: 0]"),
        "G7 trailer must survive in the full copy:\n{fetched:.200}"
    );
    assert!(
        fetched.contains("\n1000"),
        "tail of output lost:\n{fetched:.200}"
    );
    assert!(
        fetched.contains("1\n"),
        "head of output lost:\n{fetched:.200}"
    );
}
