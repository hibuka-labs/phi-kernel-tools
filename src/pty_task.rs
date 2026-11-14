//! PTY background task supervisor.
//!
//! Runs a [`LocalPty`] task inside the background-task protocol: pumps
//! detection over the screen tail (the same dual condition as the foreground
//! path — tail matches **and** output quiet ≥ `prompt_quiet_ms`), records the
//! waiting-input state the `task_output` early return reads, and fires the
//! detection notice. Timeout and cancel kill the child's process group, like
//! every background task.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

use crate::background_shell::{BackgroundTaskRegistry, BackgroundTaskStatus};
use crate::interaction::{PromptClass, TAIL_CHARS, WaitingPrompt, match_prompt, strip_ansi};
use crate::term_session::LocalPty;

/// Where a detection notice goes (TUI visibility). The product's build wires
/// this to the run's user-event channel; tests capture it directly.
pub type PromptNoticeSink = Arc<dyn Fn(&PromptClass, &str) + Send + Sync>;

/// How often the supervisor polls for child exit (the PTY has no "child
/// exited" channel of its own; reader EOF plus this poll reap it).
const EXIT_POLL_MS: u64 = 250;

/// Supervise one PTY task until it exits, times out, or is cancelled.
pub(crate) async fn run_pty_background_task(
    registry: &BackgroundTaskRegistry,
    task_id: &str,
    session: Arc<LocalPty>,
    timeout_ms: u64,
    cancel_token: CancellationToken,
    notice: PromptNoticeSink,
    prompt_quiet_ms: u64,
) {
    let screen = session.screen();
    let mut screen_rx = screen.subscribe();

    // timeout_ms == 0 → run indefinitely: no fuse.
    let unlimited = timeout_ms == 0;
    let sleep = tokio::time::sleep(Duration::from_millis(if unlimited {
        u64::MAX
    } else {
        timeout_ms
    }));
    tokio::pin!(sleep);
    let mut exit_poll = tokio::time::interval(Duration::from_millis(EXIT_POLL_MS));
    exit_poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // Detection state: the last matched class/tail and when its quiet window
    // elapses. A change event re-arms the deadline (output is not quiet).
    let mut armed: Option<(PromptClass, String, Instant)> = None;
    // Whether the registry currently holds the waiting-input marker.
    let mut waiting_set = false;

    loop {
        tokio::select! {
            changed = screen_rx.changed() => {
                // A dropped sender or an EOF bump both land here; the exit
                // poll settles the outcome either way.
                let _ = changed;
                if waiting_set {
                    // Output resumed: the quiet half of the dual condition no
                    // longer holds — the prompt is answered or the program
                    // moved on.
                    registry.clear_waiting_input(task_id);
                    waiting_set = false;
                }
                armed = evaluate_tail(&screen, prompt_quiet_ms);
            }
            _ = tokio::time::sleep_until(
                tokio::time::Instant::from_std(
                    armed.as_ref().map(|(_, _, deadline)| *deadline)
                        .unwrap_or_else(|| Instant::now() + Duration::from_secs(3600))
                )
            ), if armed.is_some() => {
                let (class, tail, _) = armed.take().expect("armed deadline implies a state");
                registry.set_waiting_input(
                    task_id,
                    WaitingPrompt { class, tail: tail.clone() },
                );
                waiting_set = true;
                notice(&class, &tail);
            }
            _ = exit_poll.tick() => {
                if let Some(code) = session.try_wait_exit() {
                    registry.finish(task_id, Some(code));
                    return;
                }
            }
            _ = &mut sleep, if !unlimited => {
                session.kill();
                tracing::warn!(task_id, timeout_ms, "pty task: timed out");
                registry.update_status(task_id, BackgroundTaskStatus::TimedOut);
                return;
            }
            _ = cancel_token.cancelled() => {
                session.kill();
                tracing::info!(task_id, "pty task: cancelled");
                registry.update_status(task_id, BackgroundTaskStatus::Cancelled);
                return;
            }
        }
    }
}

/// Re-evaluate the screen tail: if it matches a prompt, arm the quiet window
/// (deadline = last output + `prompt_quiet_ms`); otherwise nothing is armed.
fn evaluate_tail(
    screen: &crate::term_session::ScreenBuffer,
    prompt_quiet_ms: u64,
) -> Option<(PromptClass, String, Instant)> {
    let tail = strip_ansi(&screen.read_tail(TAIL_CHARS));
    let class = match_prompt(&tail)?;
    let deadline = screen.last_write() + Duration::from_millis(prompt_quiet_ms);
    Some((class, tail, deadline))
}

#[cfg(test)]
#[path = "pty_task_tests.rs"]
mod tests;
