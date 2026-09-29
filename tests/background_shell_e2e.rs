//! Phase 4 end-to-end integration tests for background shell execution.
//!
//! Uses real LocalShellTool + TaskOutputTool + TaskCancelTool with a shared
//! BackgroundTaskRegistry and ToolContext::for_test().

use std::sync::Arc;
use std::time::Duration;

use agent_base::{Tool, ToolContext, tool::content_text};
use phi_kernel_tools::background_shell::{BackgroundTaskRegistry, TaskCancelTool, TaskOutputTool};
use phi_kernel_tools::local_shell::LocalShellTool;
use serde_json::{Value, json};

fn ctx() -> ToolContext {
    ToolContext::for_test()
}

/// Helper: create registry + all 3 tools wired together.
fn setup(
    max_tasks: usize,
    shell_timeout_ms: u64,
) -> (
    Arc<BackgroundTaskRegistry>,
    LocalShellTool,
    TaskOutputTool,
    TaskCancelTool,
) {
    let reg = BackgroundTaskRegistry::new(max_tasks);
    let shell = LocalShellTool::new(shell_timeout_ms).with_registry(reg.clone());
    let output = TaskOutputTool::new(reg.clone());
    let cancel = TaskCancelTool::new(reg.clone());
    (reg, shell, output, cancel)
}

fn parse_json(text: &str) -> Value {
    serde_json::from_str(text).expect("response should be valid JSON")
}

// ── Test 1: background + foreground concurrent ─────────────────────

#[tokio::test]
async fn background_and_foreground_both_succeed() {
    let (_reg, shell, out, _cancel) = setup(4, 10_000);

    // Start background command
    let bg_result = shell
        .call(
            &json!({"command": "sleep 0.3 && echo bg_done", "background": true}),
            &ctx(),
        )
        .await
        .unwrap();
    let bg_text = content_text(&bg_result);
    let bg_json = parse_json(&bg_text);
    assert_eq!(bg_json["background"], true);
    let tid = bg_json["task_id"].as_str().unwrap().to_string();

    // While bg runs, run a foreground command
    let fg_result = shell
        .call(&json!({"command": "echo fg_hello"}), &ctx())
        .await
        .unwrap();
    let fg_text = content_text(&fg_result);
    assert!(
        fg_text.contains("fg_hello"),
        "foreground should work: {fg_text}"
    );

    // Wait for bg to finish
    let done_text = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 5000}),
            &ctx(),
        )
        .await
        .unwrap();
    let done_json = parse_json(&content_text(&done_text));
    assert_eq!(done_json["status"], "done");
    assert_eq!(done_json["exit_code"], 0);
    assert!(
        done_json["stdout"].as_str().unwrap().contains("bg_done"),
        "bg stdout should contain 'bg_done'"
    );
}

// ── Test 2: background timeout ─────────────────────────────────────

#[tokio::test]
async fn background_task_times_out() {
    // Explicit per-call fuse: background jobs default to indefinite
    // (daemon-style), so the test must arm its own short timeout.
    let (_reg, shell, out, _cancel) = setup(4, 200);

    let bg_result = shell
        .call(
            &json!({"command": "sleep 30", "background": true, "timeout_ms": 200}),
            &ctx(),
        )
        .await
        .unwrap();
    let bg_json = parse_json(&content_text(&bg_result));
    let tid = bg_json["task_id"].as_str().unwrap().to_string();

    // Wait for the timeout to fire
    let snap_text = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 5000}),
            &ctx(),
        )
        .await
        .unwrap();
    let snap_json = parse_json(&content_text(&snap_text));
    assert_eq!(
        snap_json["status"],
        "timed_out",
        "expected timed_out: {}",
        content_text(&snap_text)
    );
}

// ── Test 3: cancel kills process group ─────────────────────────────

#[tokio::test]
async fn cancel_kills_background_process_group() {
    let (reg, shell, out, cancel) = setup(4, 30_000);

    // Start a long-lived background process
    let bg_result = shell
        .call(&json!({"command": "sleep 100", "background": true}), &ctx())
        .await
        .unwrap();
    let tid = parse_json(&content_text(&bg_result))["task_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Give it a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Cancel it
    let cancel_text = cancel
        .call(&json!({"task_id": &tid}), &ctx())
        .await
        .unwrap();
    assert!(content_text(&cancel_text).contains("cancelled"));

    // Verify terminal state
    let snap = reg.snapshot(&tid).unwrap();
    assert_eq!(
        snap.status,
        phi_kernel_tools::background_shell::BackgroundTaskStatus::Cancelled
    );

    // task_output should also report cancelled
    let out_text = out
        .call(&json!({"task_id": &tid, "wait": false}), &ctx())
        .await
        .unwrap();
    assert!(content_text(&out_text).contains("\"cancelled\""));
}

// ── Test 4: wait on already-completed task returns immediately ──────

#[tokio::test]
async fn wait_on_completed_task_returns_immediately() {
    let (_reg, shell, out, _cancel) = setup(4, 10_000);

    // Start a quick command
    let bg_result = shell
        .call(
            &json!({"command": "echo instant", "background": true}),
            &ctx(),
        )
        .await
        .unwrap();
    let tid = parse_json(&content_text(&bg_result))["task_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Wait for it to finish
    let done_text = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 5000}),
            &ctx(),
        )
        .await
        .unwrap();
    let done_json = parse_json(&content_text(&done_text));
    assert_eq!(done_json["status"], "done");

    // Second wait should return immediately (already done)
    let start = std::time::Instant::now();
    let again_text = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 5000}),
            &ctx(),
        )
        .await
        .unwrap();
    let elapsed = start.elapsed();
    let again_json = parse_json(&content_text(&again_text));
    assert_eq!(again_json["status"], "done");
    assert!(
        elapsed < Duration::from_secs(1),
        "wait on completed task should return immediately, took {:?}",
        elapsed
    );
}

// ── Test 5: max_background_tasks error ─────────────────────────────

#[tokio::test]
async fn exceeding_max_tasks_returns_error() {
    let (_reg, shell, _out, _cancel) = setup(2, 30_000);

    // Fill both slots
    let _ = shell
        .call(&json!({"command": "sleep 30", "background": true}), &ctx())
        .await
        .unwrap();
    let _ = shell
        .call(&json!({"command": "sleep 30", "background": true}), &ctx())
        .await
        .unwrap();

    // Third should fail
    let result = shell
        .call(
            &json!({"command": "echo overflow", "background": true}),
            &ctx(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(
        text.contains("too many"),
        "expected 'too many' error: {text}"
    );
}

// ── Test 6: GC'd task returns not_found ────────────────────────────

#[tokio::test]
async fn gced_task_returns_not_found() {
    let (reg, shell, out, _cancel) = setup(4, 10_000);

    // Start and wait for completion
    let bg_result = shell
        .call(
            &json!({"command": "echo gc_test", "background": true}),
            &ctx(),
        )
        .await
        .unwrap();
    let tid = parse_json(&content_text(&bg_result))["task_id"]
        .as_str()
        .unwrap()
        .to_string();

    let done_text = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 5000}),
            &ctx(),
        )
        .await
        .unwrap();
    assert!(content_text(&done_text).contains("\"done\""));

    // Force GC with 0 TTL (consumed by the wait:true above → swept).
    reg.gc(Duration::ZERO);

    // Query should return not_found
    let not_found_text = out.call(&json!({"task_id": &tid}), &ctx()).await.unwrap();
    assert!(
        content_text(&not_found_text).contains("not_found"),
        "expected not_found after GC: {}",
        content_text(&not_found_text)
    );
}

// ── Test 6b: multi-task burst — delayed retrieval beats GC ─────────
//
// Issue #35: outputs of a finished burst must stay retrievable until
// *consumed* by task_output; a delayed reader must never find them gone.
// Consumption — not the clock — makes an entry GC-eligible.

#[tokio::test]
async fn burst_delayed_retrieval_full_output_retrievable() {
    let (reg, shell, out, _cancel) = setup(8, 10_000);

    // N tasks finish in a burst; nobody fetches yet.
    let mut ids = Vec::new();
    for i in 0..3 {
        let bg_result = shell
            .call(
                &json!({"command": format!("echo burst_{i}"), "background": true}),
                &ctx(),
            )
            .await
            .unwrap();
        ids.push(
            parse_json(&content_text(&bg_result))["task_id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    while reg.running_count() > 0 {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // A sweep long after the old 5-min TTL would have fired keeps
    // everything: nothing was consumed yet (AC2).
    reg.gc(Duration::from_secs(1800));
    assert_eq!(reg.snapshot_all().len(), 3, "unconsumed burst survives");

    // One fetch consumes exactly that task (AC2: swept on the next gc)…
    let first = out.call(&json!({"task_id": &ids[0]}), &ctx()).await.unwrap();
    let first_text = content_text(&first);
    assert!(
        first_text.contains("burst_0") && first_text.contains("\"done\""),
        "full output for task 0: {first_text}"
    );
    reg.gc(Duration::from_secs(1800));
    let live: Vec<_> = reg.snapshot_all().into_iter().map(|s| s.id).collect();
    assert!(!live.contains(&ids[0]), "consumed task swept");
    assert_eq!(live.len(), 2, "unconsumed siblings kept");

    // …while the delayed reader of the others still gets the full output (AC1).
    for (i, tid) in ids.iter().enumerate().skip(1) {
        let text = content_text(&out.call(&json!({"task_id": tid}), &ctx()).await.unwrap());
        assert!(
            text.contains(&format!("burst_{i}")) && text.contains("\"done\""),
            "delayed full output for task {i}: {text}"
        );
    }

    // Fallback TTL collects whatever was never fetched (AC2, second half).
    let orphan = shell
        .call(
            &json!({"command": "echo never_fetched", "background": true}),
            &ctx(),
        )
        .await
        .unwrap();
    let oid = parse_json(&content_text(&orphan))["task_id"]
        .as_str()
        .unwrap()
        .to_string();
    while reg.running_count() > 0 {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    reg.gc(Duration::ZERO);
    let text = content_text(&out.call(&json!({"task_id": &oid}), &ctx()).await.unwrap());
    assert!(
        text.contains("not_found"),
        "unconsumed but past fallback TTL: {text}"
    );
}

// ── Test 7: bad working_dir → Error ────────────────────────────────

#[tokio::test]
async fn bad_working_dir_sets_error_status() {
    let (reg, shell, out, _cancel) = setup(4, 10_000);

    let bg_result = shell
        .call(
            &json!({"command": "echo fail", "background": true, "working_dir": "/nonexistent_path_xyz"}),
            &ctx(),
        )
        .await
        .unwrap();
    let tid = parse_json(&content_text(&bg_result))["task_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Wait for the background task to attempt spawn
    tokio::time::sleep(Duration::from_millis(300)).await;

    let snap = reg.snapshot(&tid).unwrap();
    assert!(
        matches!(
            snap.status,
            phi_kernel_tools::background_shell::BackgroundTaskStatus::Error(_)
        ),
        "expected Error status, got {:?}",
        snap.status
    );

    // task_output should also report the error
    let out_text = out.call(&json!({"task_id": &tid}), &ctx()).await.unwrap();
    assert!(
        content_text(&out_text).contains("\"error\""),
        "expected error in output: {}",
        content_text(&out_text)
    );
}

// ── Test 8: output streaming ───────────────────────────────────────

#[tokio::test]
async fn background_output_is_captured() {
    let (_reg, shell, out, _cancel) = setup(4, 10_000);

    let bg_result = shell
        .call(
            &json!({"command": "echo line1 && echo line2 && echo line3", "background": true}),
            &ctx(),
        )
        .await
        .unwrap();
    let tid = parse_json(&content_text(&bg_result))["task_id"]
        .as_str()
        .unwrap()
        .to_string();

    let done_text = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 5000}),
            &ctx(),
        )
        .await
        .unwrap();
    let done_json = parse_json(&content_text(&done_text));
    assert_eq!(done_json["status"], "done");
    let stdout = done_json["stdout"].as_str().unwrap();
    assert!(stdout.contains("line1"), "stdout missing line1: {stdout}");
    assert!(stdout.contains("line2"), "stdout missing line2: {stdout}");
    assert!(stdout.contains("line3"), "stdout missing line3: {stdout}");
}
