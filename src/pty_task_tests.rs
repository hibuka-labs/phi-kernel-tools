//! PTY supervisor tests: the detection loop's two observable contracts —
//! the notice fires within ~1s of the prompt going quiet (event order:
//! registry marker first, notice second), and `wait_input_or_terminal` returns
//! early on a blocked task instead of burning the wait timeout.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

use super::run_pty_background_task;
use crate::background_shell::{BackgroundTaskRegistry, BackgroundTaskStatus};
use crate::interaction::{PromptClass, PromptKind};
use crate::term_session::{LocalPty, TermSession};

/// A prompt that blocks forever — detection must fire without any answer.
const FIXTURE_STUCK: &str = r#"printf 'Continue? [Y/n] '; sleep 30"#;

/// A prompt that completes when answered — the input round-trip fixture.
const FIXTURE_ASK: &str = r#"printf 'Continue? [Y/n] '; read ans; echo got:$ans"#;

type NoticeLog = Arc<Mutex<Vec<(PromptClass, String)>>>;

fn notice_sink(log: &NoticeLog) -> crate::pty_task::PromptNoticeSink {
    let log = Arc::clone(log);
    Arc::new(move |class: &PromptClass, tail: &str| {
        log.lock().unwrap().push((*class, tail.to_string()));
    })
}

/// Spawn the supervisor over a fresh PTY fixture; returns (registry, id,
/// session, notice log).
fn spawn_supervised(
    fixture: &str,
    timeout_ms: u64,
    prompt_quiet_ms: u64,
) -> (
    Arc<BackgroundTaskRegistry>,
    String,
    Arc<LocalPty>,
    NoticeLog,
) {
    let registry = BackgroundTaskRegistry::new(8);
    let session = LocalPty::open(fixture, None, 80, 24).expect("open pty");
    let cancel = CancellationToken::new();
    let id = registry
        .register_pty(
            fixture,
            None,
            cancel.clone(),
            session.pid(),
            timeout_ms,
            session.clone(),
        )
        .expect("register");
    let log: NoticeLog = Arc::new(Mutex::new(Vec::new()));
    {
        let reg = Arc::clone(&registry);
        let tid = id.clone();
        let sess = Arc::clone(&session);
        let notice = notice_sink(&log);
        tokio::spawn(async move {
            run_pty_background_task(
                &reg,
                &tid,
                sess,
                timeout_ms,
                cancel,
                notice,
                prompt_quiet_ms,
            )
            .await;
        });
    }
    (registry, id, session, log)
}

/// protects: the detection notice contract — after a prompt tail matches and
///   output goes quiet, the notice fires fast (≤ ~1s of the quiet window) and
///   the registry's waiting marker is already set when it fires.
/// fails_when: the notice is late (a blocked task stays invisible past the
///   quiet window), silent, or fires before the marker exists.
/// no_claim: what the product does with the notice.
#[tokio::test]
async fn notice_fires_fast_after_quiet_match() {
    let started = Instant::now();
    let (registry, id, _session, log) = spawn_supervised(FIXTURE_STUCK, 2_500, 200);

    // Wait for the notice (the fixture never answers, so this can only come
    // from detection).
    let notice_at = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if !log.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await;
    assert!(notice_at.is_ok(), "notice never fired within 2s");
    let since_start = started.elapsed();
    assert!(
        since_start < Duration::from_millis(1_000),
        "notice too slow: {since_start:?} (acceptance: ≤1s; quiet window is 200ms)"
    );

    let entries = log.lock().unwrap().clone();
    assert_eq!(
        entries.len(),
        1,
        "exactly one notice per detection: {entries:?}"
    );
    assert_eq!(entries[0].0.kind, PromptKind::Confirm);
    assert!(
        entries[0].1.contains("[Y/n]"),
        "tail echoed: {}",
        entries[0].1
    );

    // Event order: the waiting marker is set before the notice fires, so it
    // must be observable here.
    let snap = registry.snapshot(&id).expect("live task");
    let waiting = snap.waiting_input.expect("waiting marker set");
    assert_eq!(waiting.class.kind, PromptKind::Confirm);
    assert!(
        waiting.tail.contains("[Y/n]"),
        "tail recorded: {}",
        waiting.tail
    );

    // The supervisor's timeout fuse reaps the stuck fixture (no leaked sleep).
    let done = registry.wait(&id, Duration::from_secs(4)).await;
    assert!(
        matches!(done.map(|s| s.status), Some(BackgroundTaskStatus::TimedOut)),
        "timeout fuse must kill the stuck task"
    );
}

/// protects: the `task_output(wait: true)` early-return contract — a task
///   blocked on a prompt returns its snapshot within the quiet window, well
///   before the wait timeout, and an answer through the session releases it
///   to a clean exit.
/// fails_when: wait burns the full timeout on a blocked task, the marker is
///   missing, or a sent answer does not unblock the task.
/// no_claim: the tool-level wire (that is the E2E test's job).
#[tokio::test]
async fn wait_returns_early_on_prompt_and_input_releases_it() {
    let (registry, id, session, _log) = spawn_supervised(FIXTURE_ASK, 8_000, 200);

    let started = Instant::now();
    let snap = registry
        .wait_input_or_terminal(&id, Duration::from_secs(10))
        .await
        .expect("snapshot");
    let waited = started.elapsed();
    assert!(
        waited < Duration::from_secs(3),
        "wait must return early on a blocked task, took {waited:?}"
    );
    assert!(
        snap.waiting_input.is_some(),
        "early return is keyed on the waiting marker"
    );

    // Feed the prompt exactly like `task_input` does; the task must finish.
    session.send_text("y\n").await.expect("send");
    let done = registry
        .wait(&id, Duration::from_secs(5))
        .await
        .expect("terminal snapshot");
    assert_eq!(done.status, BackgroundTaskStatus::Done);
    assert_eq!(done.exit_code, Some(0));
    let screen = session.read_screen(4_096).await;
    assert!(
        screen.contains("got:y"),
        "answer reached the program: {screen}"
    );
}
