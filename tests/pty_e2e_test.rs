//! PTY task surface end-to-end: `execute_command(pty:true)` → `task_output`
//! screen + `[waiting for input]` early return → `task_input` answers → the
//! task exits 0. Spec acceptance #6, driven through the real tools.

use agent_base::tool::content_text;
use agent_base::{Tool, ToolContext};
use phi_kernel_tools::background_shell::{BackgroundTaskRegistry, TaskOutputTool};
use phi_kernel_tools::interaction::policy::{PolicyTier, PromptPolicy};
use phi_kernel_tools::local_shell::LocalShellTool;
use phi_kernel_tools::task_input::TaskInputTool;
use serde_json::{Value, json};

/// A prompt that only completes when answered — the round-trip fixture.
const FIXTURE_ASK: &str = r#"printf 'Continue? [Y/n] '; read ans; echo got:$ans"#;

fn ctx() -> ToolContext {
    ToolContext::for_test()
}

fn parse_json(text: &str) -> Value {
    serde_json::from_str(text).expect("response should be valid JSON")
}

/// All-Auto table: this test exercises the plumbing, not the consent tiers
/// (those are pinned by `task_input`'s unit tests).
fn direct_policy() -> PromptPolicy {
    PromptPolicy {
        pager: PolicyTier::Auto,
        confirm: PolicyTier::Auto,
        menu: PolicyTier::Auto,
        press_enter: PolicyTier::Auto,
        free_input: PolicyTier::Auto,
    }
}

/// protects: the PTY round trip — a blocked interactive task is observed as
///   `[waiting for input]`, answered through `task_input`, and finishes with
///   the answer's effect visible on the screen the envelope reports.
/// fails_when: any leg drops (no early return, the answer never reaches the
///   child, the envelope hides the screen, the exit code is wrong).
/// no_claim: policy tiers beyond the direct-send path.
#[tokio::test]
async fn pty_prompt_round_trip_through_the_tools() {
    let reg = BackgroundTaskRegistry::new(8);
    let shell = LocalShellTool::new(30_000)
        .with_registry(reg.clone())
        .with_prompt_quiet(200);
    let out = TaskOutputTool::new(reg.clone());
    let input = TaskInputTool::new(reg.clone()).with_policy(direct_policy());

    // 1. Start the interactive task under a PTY.
    let started = shell
        .call(
            &json!({
                "command": FIXTURE_ASK,
                "background": true,
                "pty": true,
                "timeout_ms": 8000
            }),
            &ctx(),
        )
        .await
        .unwrap();
    let started_json = parse_json(&content_text(&started));
    assert_eq!(started_json["background"], true);
    assert_eq!(started_json["pty"], true);
    let tid = started_json["task_id"].as_str().unwrap().to_string();

    // 2. The prompt blocks the task: `wait: true` returns the waiting block
    //    early instead of burning the timeout.
    let waiting = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 4000}),
            &ctx(),
        )
        .await
        .unwrap();
    let waiting_text = content_text(&waiting);
    assert!(
        waiting_text.starts_with("[waiting for input]"),
        "expected the waiting block, got: {waiting_text}"
    );
    assert!(
        waiting_text.contains("class: confirm"),
        "got: {waiting_text}"
    );
    assert!(
        waiting_text.contains("task_input"),
        "the block must name the answering tool: {waiting_text}"
    );

    // 3. Answer it — the bytes must reach the child.
    let sent = input
        .call(&json!({"task_id": &tid, "text": "y\n"}), &ctx())
        .await
        .unwrap();
    assert!(
        content_text(&sent).contains("sent"),
        "send receipt: {}",
        content_text(&sent)
    );

    // 4. The task completes with the answer's effect on the screen the
    //    envelope reports (PTY stdout face = live terminal screen).
    let done = out
        .call(
            &json!({"task_id": &tid, "wait": true, "timeout_ms": 5000}),
            &ctx(),
        )
        .await
        .unwrap();
    let done_json = parse_json(&content_text(&done));
    assert_eq!(done_json["status"], "done");
    assert_eq!(done_json["exit_code"], 0);
    let stdout = done_json["stdout"].as_str().unwrap_or("");
    assert!(
        stdout.contains("got:y"),
        "answer reached the program: {stdout}"
    );
}

/// protects: the surface guards — `pty` is background-only, and only PTY
///   tasks can receive input.
/// fails_when: a foreground `pty: true` starts, or `task_input` feeds a plain
///   pipe task.
/// no_claim: error wording beyond the fact and the fix.
#[tokio::test]
async fn pty_surface_guards() {
    let reg = BackgroundTaskRegistry::new(8);
    let shell = LocalShellTool::new(30_000).with_registry(reg.clone());
    let input = TaskInputTool::new(reg.clone()).with_policy(direct_policy());

    // Foreground + pty is rejected before anything spawns.
    let rejected = shell
        .call(&json!({"command": "echo hi", "pty": true}), &ctx())
        .await
        .unwrap();
    assert!(
        content_text(&rejected).contains("pty: true requires background: true"),
        "got: {}",
        content_text(&rejected)
    );

    // A plain pipe task cannot receive input.
    let started = shell
        .call(&json!({"command": "sleep 2", "background": true}), &ctx())
        .await
        .unwrap();
    let tid = parse_json(&content_text(&started))["task_id"]
        .as_str()
        .unwrap()
        .to_string();
    let refused = input
        .call(&json!({"task_id": &tid, "text": "y\n"}), &ctx())
        .await
        .unwrap();
    assert!(
        content_text(&refused).contains("not a PTY task"),
        "got: {}",
        content_text(&refused)
    );

    // Cleanup: stop the sleeper.
    reg.cancel(&tid);
}
