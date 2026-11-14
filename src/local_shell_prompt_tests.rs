//! Tests for [`super`] — foreground prompt detection: early-kill on
//! "tail matches && output quiet", timeout annotation, and the zero-regression
//! snapshot when nothing matches (split out of `local_shell.rs` to keep every
//! source file under the 1000-line rule; `#[path]` keeps private-item access).

use super::*;
use agent_base::tool::content_text;

/// protects: a command blocked on a no-newline prompt returns the
///   `[waiting for input]` block (class + tail + suggestion) instead of
///   hanging to the 120s fuse, and the process group is killed.
/// fails_when: the call runs to timeout, the block loses a field, or the
///   killed process leaks.
/// no_claim: what input should be sent (policy), or PTY task behavior (P1).
#[tokio::test]
async fn prompt_early_exit_prints_block() {
    let tool = LocalShellTool::new(30000);
    let start = std::time::Instant::now();
    let result = tool
        .call(
            &json!({"command": "printf 'Continue? [Y/n] '; sleep 30"}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(text.starts_with("[waiting for input]"), "got: {text}");
    assert!(text.contains("class: confirm"));
    assert!(text.contains("default: y"));
    assert!(text.contains("suggested: y\\n"));
    assert!(text.contains("tail:\nContinue? [Y/n]"), "got: {text}");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "must early-exit well before the tool fuse, took {:?}",
        start.elapsed()
    );
}

/// protects: a password prompt early-exits with the password class and a
///   never-send suggestion at every policy tier.
/// fails_when: the class is not `password` or any suggestion is attached.
/// no_claim: the human handoff UX.
#[tokio::test]
async fn prompt_early_exit_password() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(
            &json!({"command": "printf 'Password: '; sleep 30"}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(text.starts_with("[waiting for input]"), "got: {text}");
    assert!(text.contains("class: password"));
    assert!(text.contains("suggested: none"));
}

/// protects: the quiet window is armed at the default `PROMPT_QUIET_MS` —
///   no early exit at 2.9s of quiet, early exit right after 3s.
/// fails_when: the kill fires before the quiet window elapses (false positive)
///   or never fires (back to the bare timeout).
/// no_claim: the exact wall-clock fire time (chunk arrival adds latency).
#[tokio::test]
async fn prompt_quiet_boundary() {
    let tool = LocalShellTool::new(30000);
    let ctx = ToolContext::for_test();
    let args = json!({"command": "printf 'Continue? [Y/n] '; sleep 30"});
    let fut = tool.call(&args, &ctx);
    tokio::pin!(fut);
    let start = std::time::Instant::now();
    let early = tokio::time::timeout(Duration::from_millis(2900), &mut fut).await;
    assert!(
        early.is_err(),
        "must not fire before the quiet window elapses"
    );
    let done = tokio::time::timeout(Duration::from_millis(2500), &mut fut).await;
    let result = done
        .expect("early exit fires after the quiet window plus spawn slack")
        .unwrap();
    assert!(
        content_text(&result).starts_with("[waiting for input]"),
        "quiet boundary fired at {:?}",
        start.elapsed()
    );
}

/// protects: the quiet half of the dual condition — a run that keeps producing
///   output is never killed, even with a prompt-shaped line in its tail.
/// fails_when: the early exit fires while output is still flowing.
/// no_claim: detection of prompts that appear only between quiet periods.
#[tokio::test]
async fn quiet_reset_by_new_output() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(
            &json!({"command": "for i in 1 2 3 4 5 6 7 8 9 10; do printf 'Continue? [Y/n] step-'$i'\\n'; sleep 0.5; done"}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(!text.contains("[waiting for input]"), "got: {text}");
    assert!(text.contains("[exit code: 0]"), "got: {text}");
}

/// protects: a prompt-shaped tail that never goes quiet still gets the
///   `[waiting for input]` annotation when the fuse fires.
/// fails_when: the timeout message is bare (annotation missing) or the
///   no-match timeout message shape drifts (see the snapshot card).
/// no_claim: what the command was actually blocked on.
#[tokio::test]
async fn timeout_branch_annotation() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(
            &json!({
                "command": "printf 'Continue? [Y/n]\\n'; while true; do printf 'Continue? [Y/n] still working\\n'; sleep 0.3; done",
                "timeout_ms": 1200
            }),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(
        text.starts_with("[Command Timed Out after 1200ms]\ncommand: "),
        "got: {text}"
    );
    assert!(text.contains("[waiting for input]"), "got: {text}");
    assert!(text.contains("class: confirm"), "got: {text}");
}

/// protects: zero regression — with no prompt match the success result is
///   byte-identical to the pre-detection format.
/// fails_when: any byte of the result differs from `hello\n[exit code: 0]`.
/// no_claim: the annotation paths.
#[tokio::test]
async fn no_match_zero_regression() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(&json!({"command": "echo hello"}), &ToolContext::for_test())
        .await
        .unwrap();
    assert_eq!(content_text(&result), "hello\n[exit code: 0]");
}

/// protects: zero regression on the timeout arm — with no prompt match the
///   timeout message is byte-identical to the pre-detection format.
/// fails_when: the message gains or loses any byte.
/// no_claim: the annotated timeout arm.
#[tokio::test]
async fn timeout_no_match_unchanged() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(
            &json!({"command": "sleep 5", "timeout_ms": 300}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    assert_eq!(
        content_text(&result),
        "[Command Timed Out after 300ms]\ncommand: sleep 5\n[hint] Foreground sleep-polling burns a whole turn and blocks the conversation. Start the long job with background=true and let its completion notice wake you; read progress with task_output."
    );
}

/// protects: a prompt split across partial writes (no newline anywhere)
///   still assembles and early-exits — the flushed-partial channel contract.
/// fails_when: the half-line flush is dropped, re-joined with a spurious
///   newline, or the match never fires.
/// no_claim: prompts printed all at once (covered above).
#[tokio::test]
async fn partial_line_prompt_assembles() {
    let tool = LocalShellTool::new(30000);
    let result = tool
        .call(
            &json!({"command": "printf 'Contin'; sleep 0.4; printf 'ue? [Y/n] '; sleep 30"}),
            &ToolContext::for_test(),
        )
        .await
        .unwrap();
    let text = content_text(&result);
    assert!(text.starts_with("[waiting for input]"), "got: {text}");
    assert!(text.contains("tail:\nContinue? [Y/n]"), "got: {text}");
}

/// protects: the false-positive defense — non-interactive output (including
///   prose that merely mentions prompt shapes) never early-exits; the quiet
///   half of the dual condition carries the precision.
/// fails_when: any fixture result contains `[waiting for input]`.
/// no_claim: interactive prompts (covered above), stderr-only prompts.
#[tokio::test]
async fn false_positive_fixtures_never_early_exit() {
    let fixtures: &[&str] = &[
        "printf 'Pulling from library/nginx\\nDigest: sha256:abc123\\nStatus: Downloaded newer image\\n'",
        "printf 'npm warn deprecated foo@1.0.0\\nadded 269 packages in 9s\\n'",
        "printf 'Get:1 http://deb.debian.org bullseye InRelease\\nFetched 11.2 MB in 3s\\n'",
        "printf 'Collecting requests\\nInstalling collected packages: requests\\nSuccessfully installed requests-2.31.0\\n'",
        "printf 'Compiling phi-kernel-tools v0.13.0\\nFinished dev target in 12.4s\\n'",
        "printf 'main.c:10:5: warning: unused variable x\\nmain.c:22:1: note: declared here\\n'",
        "printf 'Cloning into repo...\\nReceiving objects: 100%% (20/20), done.\\n'",
        "printf 'mkdir -p build\\ncc -o app main.c\\nstrip app\\n'",
        "printf 'Loaded plugins: fastestmirror\\nResolving Dependencies\\nComplete!\\n'",
        "printf 'Connecting to example.com...\\nHTTP request sent, awaiting response... 200 OK\\n'",
        "printf 'Apply complete! Resources: 3 added, 0 changed, 0 destroyed.\\n'",
        "printf 'NAME READY STATUS RESTARTS AGE\\nweb-1 1/1 Running 0 2d\\n'",
        "printf 'USER PID %%CPU %%MEM COMMAND\\nroot 1 0.0 0.1 /sbin/init\\n'",
        "printf 'Filesystem Size Used Avail Use%% Mounted on\\n/dev/disk1 465G 200G 265G 43%% /\\n'",
        "printf 'Mem: 16384M total 8192M used\\nSwap: 4096M total 0M used\\n'",
        "printf 'web.service - Web Server\\nActive: active (running)\\n'",
        "printf 'Nov 11 10:00:00 host systemd[1]: Started Session 42.\\n'",
        "printf 'Traceback (most recent call last):\\n  File app.py line 10\\nValueError: boom\\n'",
        "printf 'Exception in thread main java.lang.NullPointerException\\n    at App.main(App.java:10)\\n'",
        "printf '(node:42) DeprecationWarning: punycode module is deprecated\\n'",
        "printf 'error[E0308]: mismatched types\\n  --> src/lib.rs:10:5\\n'",
        "printf '============================= test session starts ==============================\\ncollected 12 items\\n'",
        "printf 'ok   package/app 0.012s\\nPASS\\n'",
        "printf 'Downloading https://repo.maven.org/foo.jar ... 42%% ... 100%%\\n'",
        "printf 'yarn install v1.22.19\\ninfo fsevents@2.3.2: The platform is incompatible\\n'",
        "printf 'Generating optimized autoload files\\n> Package manifest generated\\n'",
        "printf '==> Downloading https://example.com/foo.tar.gz\\n==> Pouring foo-1.0.tar.gz\\n'",
        "printf 'Progress: Downloading 1 of 5 packages\\nProgress: Downloading 5 of 5 packages\\n'",
        "printf 'README: when the installer shows [Y/n], press y\\nContinuing setup...\\nAll done.\\n'",
        "printf 'docs:\\nPassword: example123 is the sample token\\nend of doc\\n'",
        "printf 'manual page: the pager shows (END) at the bottom\\nusage end\\n'",
        "printf 'guide: pick from the menu [1/2/3] when prompted\\nstep 2 complete\\n'",
        "printf 'tutorial text: press Enter to continue is shown once per screen\\nnext section\\n'",
        "printf 'FAQ: answer (yes/no) prompts with care\\nDone\\n'",
    ];
    for command in fixtures {
        let tool = LocalShellTool::new(30000);
        let result = tool
            .call(&json!({ "command": command }), &ToolContext::for_test())
            .await
            .unwrap();
        let text = content_text(&result);
        assert!(
            !text.contains("[waiting for input]"),
            "false positive for fixture: {command}\ngot: {text}"
        );
        assert!(
            text.contains("[exit code: 0]"),
            "fixture did not run cleanly: {command}\ngot: {text}"
        );
    }
}
