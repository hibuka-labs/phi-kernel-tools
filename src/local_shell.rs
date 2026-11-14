use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use agent_base::types::NoticeKind;
use agent_base::{AgentResult, Content, Tool, ToolContext, UserEvent};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::AsyncBufReadExt;
use tokio_util::sync::CancellationToken;

use crate::background_shell::{BackgroundTaskRegistry, BackgroundTaskStatus};
use crate::interaction::{PromptClass, notice_text};
use crate::pty_task::{PromptNoticeSink, run_pty_background_task};
use crate::spill::{SpillSink, spill_marker};
use crate::term_session::LocalPty;

/// Default PTY geometry for `pty: true` tasks (what a plain terminal gives).
const PTY_COLS: u16 = 80;
const PTY_ROWS: u16 = 24;

/// Local shell command execution tool.
///
/// Executes arbitrary commands via `sh -c`, with support for timeout,
/// cancellation, and working directory. Optionally supports background
/// execution via an injected `BackgroundTaskRegistry`, and full-copy spill of
/// truncated output via an injected [`SpillSink`].
pub struct LocalShellTool {
    timeout_ms: u64,
    registry: Option<Arc<BackgroundTaskRegistry>>,
    spill: Option<Arc<dyn SpillSink>>,
    prompt_quiet_ms: u64,
}

impl LocalShellTool {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            timeout_ms,
            registry: None,
            spill: None,
            prompt_quiet_ms: crate::interaction::PROMPT_QUIET_MS,
        }
    }

    /// Inject a background task registry. Enables `background: true` support.
    pub fn with_registry(mut self, registry: Arc<BackgroundTaskRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Inject a spill sink for truncated output. When a result exceeds
    /// `max_output_chars`, the full pre-truncation summary is spilled and the
    /// truncation marker carries the artifact path; without a sink (or on a
    /// sink write failure) the plain marker stands.
    pub fn with_spill(mut self, spill: Arc<dyn SpillSink>) -> Self {
        self.spill = Some(spill);
        self
    }

    /// Override the prompt-quiet window (ms): how long the output must stay
    /// quiet after a tail matches an interactive prompt before the foreground
    /// command is killed and `[waiting for input]` is returned.
    pub fn with_prompt_quiet(mut self, ms: u64) -> Self {
        self.prompt_quiet_ms = ms;
        self
    }
}

fn format_result(
    command: &str,
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
    timed_out: bool,
) -> String {
    let stdout = stdout.trim();
    let stderr = stderr.trim();

    if timed_out {
        return format!(
            "[Command Timed Out]\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
            command,
            if stdout.is_empty() { "(empty)" } else { stdout },
            if stderr.is_empty() { "(empty)" } else { stderr },
        );
    }

    match exit_code {
        // G7: success-with-output carries a trailing `[exit code: 0]` line so
        // the model can confirm exit 0 without a verification re-run. The
        // empty-success sentence and every failure/termination arm keep their
        // exact vocabulary — only the three output arms gain the trailer.
        Some(0) => match (stdout.is_empty(), stderr.is_empty()) {
            (true, true) => "Command executed successfully with no output.".to_string(),
            (false, true) => format!("{}\n[exit code: 0]", stdout),
            (true, false) => format!("stderr:\n{}\n[exit code: 0]", stderr),
            (false, false) => format!("stdout:\n{}\n\nstderr:\n{}\n[exit code: 0]", stdout, stderr),
        },
        Some(code) => format!(
            "[Command Failed (exit code: {})]\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
            code,
            command,
            if stdout.is_empty() { "(empty)" } else { stdout },
            if stderr.is_empty() { "(empty)" } else { stderr },
        ),
        None => format!(
            "[Command Terminated]\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
            command,
            if stdout.is_empty() { "(empty)" } else { stdout },
            if stderr.is_empty() { "(empty)" } else { stderr },
        ),
    }
}

/// Extra watchdog slack (ms) the pipeline gets over the tool's own fuse, so the
/// informative `[Command Timed Out]` path wins the race against `[Tool Timeout]`.
const WATCHDOG_SLACK_MS: u64 = 5_000;

/// Guidance appended to foreground results when the command is a sleep-poll.
const SLEEP_POLL_HINT: &str = "[hint] Foreground sleep-polling burns a whole turn and blocks the \
conversation. Start the long job with background=true and let its completion notice wake you; \
read progress with task_output.";

/// True when a command's first word is `sleep` — the "wait by blocking a turn"
/// anti-pattern (`sleep 300; tail log`). Hint-only: `sleep` inside real scripts
/// stays legal, we just nudge the blocking-wait shape.
fn is_sleep_poll(command: &str) -> bool {
    // `split_whitespace` already skips leading whitespace.
    command
        .split_whitespace()
        .next()
        .is_some_and(|word| word == "sleep")
}

/// Plain truncation marker (no spill target available).
pub(crate) const MARKER: &str = "...[output truncated]";

/// Truncate a shell output string to the per-call output budget
/// (`ToolContext::max_output_chars`), keeping both the head and the tail.
///
/// Compile/test errors cluster at the *end* of a command's output, while listing
/// output is meaningful at the *start*, so both ends survive and the elided
/// middle is marked. This mirrors `read_file`/`list_files`: the tool
/// self-truncates so the engine's hard reject never fires for a noisy command.
pub(crate) fn truncate_output(s: &str, max_chars: Option<usize>) -> String {
    truncate_with_marker(s, max_chars, MARKER)
}

/// [`truncate_output`] with a caller-chosen marker line — the spill path uses
/// the marker that names the full-output artifact. Byte layout is identical.
pub(crate) fn truncate_with_marker(s: &str, max_chars: Option<usize>, marker: &str) -> String {
    let Some(max_chars) = max_chars else {
        return s.to_string();
    };
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }

    // Reserve the marker + two joining newlines; head gets 1/3, tail 2/3.
    let available = max_chars.saturating_sub(marker.chars().count() + 2);
    let head_chars = available / 3;
    let tail_chars = available - head_chars;

    let head: String = s.chars().take(head_chars).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    format!("{head}\n{marker}\n{tail}")
}

/// [`truncate_output`] that keeps the cut-away middle retrievable: when a cut
/// actually fires and a sink is injected, the full pre-truncation summary is
/// spilled (kind `"shell_output"`) and the marker carries the artifact path
/// plus a `read_file` continuation hint. Sink absent or write-fail → the plain
/// marker exactly (never a fake path).
pub(crate) fn truncate_output_spilling(
    s: &str,
    max_chars: Option<usize>,
    sink: Option<&dyn SpillSink>,
) -> String {
    let Some(max_chars) = max_chars else {
        return s.to_string();
    };
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    match spill_marker(sink, "shell_output", s) {
        Some(marker) => truncate_with_marker(s, Some(max_chars), &marker),
        None => truncate_output(s, Some(max_chars)),
    }
}

#[async_trait]
impl Tool for LocalShellTool {
    fn name(&self) -> &'static str {
        "execute_command"
    }

    fn description(&self) -> &'static str {
        "Execute a shell command locally.\n\
         - Foreground (default): blocks until the command exits, or is killed at timeout_ms (default 120000).\n\
         - Background (background=true): returns a task_id immediately and keeps running with no timeout by default; its exit sends a system notification. For a server, an exit notification means it died — investigate via task_output(task_id). Read output anytime with task_output; stop with task_cancel.\n\
         - timeout_ms: kill the process after N ms; 0 = never kill. Background defaults to 0 (indefinite), foreground to 120000. Set it only on background jobs that must finish within a bound (e.g. a long test run).\n\
         - PTY (pty=true, background only): runs under a terminal, so interactive prompts work. task_output returns `[waiting for input]` when one blocks the task; answer it with task_input. Use for programs that need a TTY (ssh, REPLs, installers, pagers).\n\
         Decide like a person: servers/watchers (vite, mvn spring-boot:run) → background; verify they listen (e.g. lsof -ti:PORT) before reporting success, then leave them alone. Long one-shot jobs (mvn test, npm install) → background with a generous timeout_ms, or foreground when quick. For large expected output, limit lines (grep ... | head -n 30).\n\
         NEVER wait by sleeping in a foreground call (`sleep 120; tail log` burns a whole turn and blocks the conversation): start the job with background=true and let its completion notice wake you, then read progress with task_output."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. For commands that may produce large output, consider limiting lines: cat large files with | tail -n 30, find / ls -R with | head -n 50, grep over large scope with | head -n 30."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory. Uses the current directory if not specified."
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background. Returns immediately with a task_id. Use task_output to check results.",
                    "default": false
                },
                "pty": {
                    "type": "boolean",
                    "description": "Run under a pseudo-terminal (requires background: true). Interactive prompts work — answer them with task_input.",
                    "default": false
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Kill the process after N ms; 0 = never kill. Default: 120000 in foreground, indefinite (0) in background.",
                    "minimum": 0
                }
            },
            "required": ["command"]
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "Execute a shell command locally; supports background execution with per-call timeout_ms (0 = run indefinitely).".to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    fn call_timeout_ms(&self, args: &Value) -> Option<u64> {
        // Mirror the per-call budget parse in `call` so the pipeline watchdog
        // covers the fuse this tool arms — and never kills a call early that
        // was explicitly promised more time (session 20261018_7844215b: a 660s
        // budget was cut at the 600s framework default). `0` means unbounded.
        //
        // A small slack keeps the tool's own fuse the one that fires, so the
        // model sees `[Command Timed Out after Nms]` instead of the blunter
        // `[Tool Timeout]` when a command actually overruns.
        let is_background = args
            .get("background")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let budget = args
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(if is_background { 0 } else { self.timeout_ms });
        Some(if budget == 0 {
            0
        } else {
            budget.saturating_add(WATCHDOG_SLACK_MS)
        })
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let command = args
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();

        if command.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: No command provided.".to_string(),
            )]);
        }

        let working_dir = args.get("working_dir").and_then(Value::as_str);

        // ── Background path ──────────────────────────────────────────
        let is_background = args
            .get("background")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        // A terminal matters only while the task keeps running between turns:
        // `pty: true` is a background-only surface.
        let is_pty = args.get("pty").and_then(Value::as_bool).unwrap_or(false);
        if is_pty && !is_background {
            return Ok(vec![Content::text(
                "[Error]: pty: true requires background: true — an interactive \
                 terminal is only useful while the task keeps running between turns."
                    .to_string(),
            )]);
        }

        // Per-call timeout; 0 = never kill. Background defaults to unlimited —
        // dev servers are legitimate long-running processes, and the old
        // hard-wired fuse turned background=true into "die at 120s". Foreground
        // keeps the tool-level default because it blocks the conversation.
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(if is_background { 0 } else { self.timeout_ms });

        tracing::info!(
            command = %command,
            timeout_ms = timeout_ms,
            background = is_background,
            "execute_command start"
        );

        if is_background {
            let Some(registry) = &self.registry else {
                return Ok(vec![Content::text(
                    "[Error]: background=true requires a BackgroundTaskRegistry. \
                     No registry was injected at build time."
                        .to_string(),
                )]);
            };

            if is_pty {
                let session = match LocalPty::open(&command, working_dir, PTY_COLS, PTY_ROWS) {
                    Ok(s) => s,
                    Err(e) => {
                        return Ok(vec![Content::text(format!(
                            "[Error]: failed to open PTY: {}",
                            e
                        ))]);
                    }
                };
                let cancel_token = CancellationToken::new();
                let task_id = match registry.register_pty(
                    &command,
                    working_dir,
                    cancel_token.clone(),
                    session.pid(),
                    timeout_ms,
                    session.clone(),
                ) {
                    Ok(id) => id,
                    Err(e) => {
                        return Ok(vec![Content::text(format!("[Error]: {}", e))]);
                    }
                };

                // Detection notice rides the run's user-event channel, tagged
                // with this tool as its source. Attention (amber), not Warning
                // (red): a prompt waiting for an answer is "look at this",
                // not "this failed".
                let notice_ctx = ctx.clone();
                let notice_tid = task_id.clone();
                let notice: PromptNoticeSink = Arc::new(move |class: &PromptClass, tail: &str| {
                    notice_ctx.emit_user_event(UserEvent::Notice {
                        kind: NoticeKind::Attention,
                        source: "tool:execute_command".to_string(),
                        text: notice_text(&notice_tid, class, tail),
                    });
                });

                let reg = registry.clone();
                let tid = task_id.clone();
                let quiet_ms = self.prompt_quiet_ms;
                tokio::spawn(async move {
                    run_pty_background_task(
                        &reg,
                        &tid,
                        session,
                        timeout_ms,
                        cancel_token,
                        notice,
                        quiet_ms,
                    )
                    .await;
                });

                return Ok(vec![Content::text(format!(
                    "{}",
                    json!({
                        "background": true,
                        "pty": true,
                        "task_id": task_id,
                        "message": "Command started under a PTY. Read the screen with task_output; answer prompts with task_input."
                    })
                ))]);
            }

            let cancel_token = CancellationToken::new();
            let task_id = match registry.register(
                &command,
                working_dir,
                cancel_token.clone(),
                None, // pgid set after spawn
                timeout_ms,
            ) {
                Ok(id) => id,
                Err(e) => {
                    return Ok(vec![Content::text(format!("[Error]: {}", e))]);
                }
            };

            let reg = registry.clone();
            let cmd = command.clone();
            let dir = working_dir.map(String::from);
            let tid = task_id.clone();

            // Spawn a 'static executor task — all data is owned/cloned in.
            tokio::spawn(async move {
                run_background_task(&reg, &tid, &cmd, dir.as_deref(), timeout_ms, cancel_token)
                    .await;
            });

            return Ok(vec![Content::text(format!(
                "{}",
                json!({
                    "background": true,
                    "task_id": task_id,
                    "message": "Command started in background. Use task_output to check."
                })
            ))]);
        }

        // ── Foreground path (unchanged) ─────────────────────────────
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(&command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            // Group cleanup is owned by `ProcessGroupGuard` (below), which kills
            // the whole tree — not just the direct child — even when the engine
            // drops this future on cancel.
            .kill_on_drop(false);

        // Run in its own process group so a timeout/cancel kills the whole tree
        // (e.g. `mvn spring-boot:run` → `java`), not just the `sh` wrapper —
        // otherwise the grandchild survives as an orphan server. With pgroup 0,
        // the child's pid doubles as its process-group id, so `kill -9 -<pid>`
        // reaches every descendant.
        #[cfg(unix)]
        cmd.process_group(0);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, command = %command, "execute_command: spawn failed");
                return Ok(vec![Content::text(format!(
                    "[Error]: Command execution failed: {}",
                    e
                ))]);
            }
        };

        // Under unix `process_group(0)` this is also the process-group id.
        let pid = child.id();

        // Take the pipes before moving `child` into the guard, so the reader
        // tasks can stream independently of it.
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Owns the child so a dropped future (engine cancel / Ctrl+C) still kills
        // the whole process group — `kill_on_drop` alone would orphan descendants.
        let mut guard = ProcessGroupGuard {
            pgid: pid,
            child,
            done: false,
        };

        // Stream stdout/stderr via reader tasks, so the caller sees live
        // progress instead of a frozen screen for the whole duration. Chunks
        // are tagged (stderr vs stdout) and carry a `complete` flag: complete
        // lines are re-joined with '\n' exactly like the line-granular path,
        // flushed partials concatenate bare — an interactive prompt almost
        // never carries a trailing newline and must reach the matcher early.
        let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<(bool, String, bool)>();

        if let Some(stdout) = stdout {
            tokio::spawn(forward_chunks(stdout, false, line_tx.clone()));
        }
        if let Some(stderr) = stderr {
            tokio::spawn(forward_chunks(stderr, true, line_tx));
        }

        let mut stdout_buf = String::new();
        let mut stderr_buf = String::new();
        // Prompt watcher state: armed when the ANSI-stripped output tail
        // matches an interactive prompt, re-evaluated on every new chunk; the
        // early exit fires only after the output has stayed quiet for
        // `prompt_quiet_ms` — the dual condition that keeps a run which merely
        // prints prompt-like text from being killed.
        let mut prompt_class: Option<crate::interaction::PromptClass> = None;
        let mut prompt_deadline: Option<tokio::time::Instant> = None;
        // timeout_ms == 0 → no fuse: disable the timeout branch and push the
        // deadline far out (from_millis(0) would fire instantly).
        let unlimited = timeout_ms == 0;
        let sleep = tokio::time::sleep(Duration::from_millis(if unlimited {
            u64::MAX
        } else {
            timeout_ms
        }));
        tokio::pin!(sleep);

        // Drive the command: drain streamed lines until both reader tasks finish
        // (the channel closes, which also implies the process exited and closed
        // its pipes), or bail on timeout/cancel. `child.wait()` is reaped after
        // the loop for the exit code.
        loop {
            tokio::select! {
                chunk = line_rx.recv() => {
                    match chunk {
                        Some((is_stderr, text, complete)) => {
                            // Live feedback; the final summary is assembled below.
                            ctx.emit_progress(&text);
                            let buf = if is_stderr { &mut stderr_buf } else { &mut stdout_buf };
                            buf.push_str(&text);
                            if complete {
                                buf.push('\n');
                            }
                            // Re-evaluate the prompt watcher on every chunk: a
                            // new hit re-arms the quiet window, a tail that no
                            // longer matches disarms it.
                            let tail = crate::interaction::output_tail(
                                &stdout_buf,
                                &stderr_buf,
                                crate::interaction::TAIL_CHARS,
                            );
                            let hit = crate::interaction::match_prompt(
                                &crate::interaction::strip_ansi(&tail),
                            );
                            prompt_deadline = hit.map(|_| {
                                tokio::time::Instant::now()
                                    + Duration::from_millis(self.prompt_quiet_ms)
                            });
                            prompt_class = hit;
                        }
                        // Both reader tasks finished → all output drained and the
                        // process exited. Reap the exit code after the loop.
                        None => break,
                    }
                }
                _ = tokio::time::sleep_until(prompt_deadline.unwrap_or_else(||
                        tokio::time::Instant::now() + Duration::from_secs(3600)
                    )), if prompt_deadline.is_some() => {
                    // Quiet window elapsed with the prompt still on the tail
                    // (any new chunk would have re-armed or cleared it).
                    kill_process_group(pid);
                    let class = prompt_class.expect("armed deadline implies a class");
                    let tail = crate::interaction::output_tail(
                        &stdout_buf,
                        &stderr_buf,
                        crate::interaction::TAIL_CHARS,
                    );
                    let block = crate::interaction::format_waiting_block(
                        &class,
                        &crate::interaction::strip_ansi(&tail),
                    );
                    tracing::warn!(
                        command = %command,
                        class = class.kind.as_str(),
                        "execute_command: tail matches an interactive prompt and output went quiet; killed process group"
                    );
                    return Ok(vec![Content::text(block)]);
                }
                _ = &mut sleep, if !unlimited => {
                    kill_process_group(pid);
                    tracing::warn!(command = %command, timeout_ms = timeout_ms, "execute_command: timed out, killed process group");
                    let mut msg = format!(
                        "[Command Timed Out after {}ms]\ncommand: {}",
                        timeout_ms, command
                    );
                    if is_sleep_poll(&command) {
                        msg.push('\n');
                        msg.push_str(SLEEP_POLL_HINT);
                    }
                    // Tail annotation: the fuse fired but the tail matches an
                    // interactive prompt — say so instead of a bare timeout.
                    // No match → the message stays byte-identical.
                    let tail = crate::interaction::output_tail(
                        &stdout_buf,
                        &stderr_buf,
                        crate::interaction::TAIL_CHARS,
                    );
                    if let Some(class) =
                        crate::interaction::match_prompt(&crate::interaction::strip_ansi(&tail))
                    {
                        msg.push('\n');
                        msg.push_str(&crate::interaction::format_waiting_block(
                            &class,
                            &crate::interaction::strip_ansi(&tail),
                        ));
                    }
                    return Ok(vec![Content::text(msg)]);
                }
                _ = ctx.cancel_token.cancelled() => {
                    kill_process_group(pid);
                    tracing::info!(command = %command, "execute_command: cancelled, killed process group");
                    return Ok(vec![Content::text(format!(
                        "[Command Terminated]\ncommand: {}",
                        command
                    ))]);
                }
            }
        }

        // Reap the child for its exit code (it already exited — its pipes closed,
        // which is what ended the loop).
        let exit_code = match guard.child.wait().await {
            Ok(status) => status.code(),
            Err(e) => {
                tracing::error!(error = %e, command = %command, "execute_command: wait failed");
                return Ok(vec![Content::text(format!(
                    "[Error]: Command execution failed: {}",
                    e
                ))]);
            }
        };
        // Reaped successfully → the guard must not kill anything on drop.
        guard.done = true;

        tracing::info!(
            command = %command,
            exit_code = exit_code,
            stdout_len = stdout_buf.len(),
            stderr_len = stderr_buf.len(),
            "execute_command: done"
        );

        let mut summary = format_result(&command, &stdout_buf, &stderr_buf, exit_code, false);
        if is_sleep_poll(&command) {
            summary.push('\n');
            summary.push_str(SLEEP_POLL_HINT);
        }
        let summary =
            truncate_output_spilling(&summary, ctx.max_output_chars, self.spill.as_deref());
        Ok(vec![Content::text(summary)])
    }
}

/// Stream a child pipe's lines into `tx`, tagged `is_stderr`, until EOF.
///
/// Output bytes are decoded **lossily** — a strict `lines()` reader used to
/// treat a decode error as EOF, which dropped the read end mid-run and made
/// the child's later writes die with EPIPE/SIGPIPE (2026-10-12 real incident:
/// under a UTF-8 locale, bash 3.2 swallows "variable value + first byte of the
/// multibyte char" when `$VAR` abuts a full-width char, so release.sh emitted
/// invalid UTF-8 and cargo test hit "BrokenPipe: listing tests").
/// Bad bytes become U+FFFD and the pipe stays open; only EOF or a real I/O
/// error ends the stream.
async fn forward_lines<R: tokio::io::AsyncRead + Unpin>(
    reader: R,
    is_stderr: bool,
    tx: tokio::sync::mpsc::UnboundedSender<(bool, String)>,
) {
    let mut reader = tokio::io::BufReader::new(reader);
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf).await {
            Ok(0) => break, // EOF — the child closed its end of the pipe.
            Ok(_) => {
                if buf.ends_with(b"\n") {
                    buf.pop();
                    if buf.ends_with(b"\r") {
                        buf.pop();
                    }
                }
                let line = String::from_utf8_lossy(&buf).into_owned();
                if tx.send((is_stderr, line)).is_err() {
                    break; // receiver gone: task finished/cancelled
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, is_stderr, "capture: pipe read failed, closing stream");
                break;
            }
        }
    }
}

/// How long a partial (newline-free) line may sit before it is flushed
/// downstream anyway. Interactive prompts ("Password: ", "[Y/n]") almost
/// never carry a trailing newline, and a pure `read_until(b'\n')` would hold
/// them back until EOF — so prompt detection would never see them.
const PARTIAL_FLUSH_MS: u64 = 150;

/// Stream a child pipe's chunks into `tx`, tagged `is_stderr`, until EOF.
/// The third field is `complete_line`: the receiver joins complete lines with
/// '\n' exactly like [`forward_lines`], and concatenates flushed partials bare
/// (a flushed partial and its later remainder are one logical line).
///
/// Cancellation-safe: `read_until` leaves whatever it consumed in `buf`, so a
/// flush-on-timeout drops no bytes. Decoding is lossy for the same reason as
/// [`forward_lines`] (a strict decoder must never read as EOF).
async fn forward_chunks<R: tokio::io::AsyncRead + Unpin>(
    reader: R,
    is_stderr: bool,
    tx: tokio::sync::mpsc::UnboundedSender<(bool, String, bool)>,
) {
    let mut reader = tokio::io::BufReader::new(reader);
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        let read = tokio::time::timeout(
            Duration::from_millis(PARTIAL_FLUSH_MS),
            reader.read_until(b'\n', &mut buf),
        );
        match read.await {
            Err(_elapsed) => {
                // No newline yet — flush what has arrived so the prompt
                // watcher can see an unterminated prompt line.
                if buf.is_empty() {
                    continue;
                }
                let text = String::from_utf8_lossy(&buf).into_owned();
                if tx.send((is_stderr, text, false)).is_err() {
                    break; // receiver gone: task finished/cancelled
                }
            }
            Ok(Ok(0)) => break, // EOF — the child closed its end of the pipe.
            Ok(Ok(_)) => {
                if buf.ends_with(b"\n") {
                    buf.pop();
                    if buf.ends_with(b"\r") {
                        buf.pop();
                    }
                }
                let text = String::from_utf8_lossy(&buf).into_owned();
                if tx.send((is_stderr, text, true)).is_err() {
                    break;
                }
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, is_stderr, "capture: pipe read failed, closing stream");
                break;
            }
        }
    }
}

/// Background executor: spawns the command, streams output to the registry,
/// and handles timeout / cancellation / normal completion.
///
/// Runs as a detached tokio task — all parameters are owned or cloned.
async fn run_background_task(
    registry: &BackgroundTaskRegistry,
    task_id: &str,
    command: &str,
    working_dir: Option<&str>,
    timeout_ms: u64,
    cancel_token: CancellationToken,
) {
    let mut cmd = tokio::process::Command::new("sh");
    cmd.arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(false);

    #[cfg(unix)]
    cmd.process_group(0);

    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, command = %command, "background: spawn failed");
            registry.set_error(task_id, format!("spawn failed: {}", e));
            return;
        }
    };

    let pid = child.id();
    registry.set_pgid(task_id, pid);

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let mut guard = ProcessGroupGuard {
        pgid: pid,
        child,
        done: false,
    };

    let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<(bool, String)>();

    if let Some(stdout) = stdout {
        tokio::spawn(forward_lines(stdout, false, line_tx.clone()));
    }
    if let Some(stderr) = stderr {
        tokio::spawn(forward_lines(stderr, true, line_tx));
    }

    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    // timeout_ms == 0 → run indefinitely: no fuse. (from_millis(0) would fire
    // instantly, so both the branch guard and the u64::MAX deadline matter.)
    let unlimited = timeout_ms == 0;
    let sleep = tokio::time::sleep(Duration::from_millis(if unlimited {
        u64::MAX
    } else {
        timeout_ms
    }));
    tokio::pin!(sleep);

    loop {
        tokio::select! {
            line = line_rx.recv() => {
                match line {
                    Some((is_stderr, line)) => {
                        // Bounded by registry's append_buffer (head 8K + tail 24K).
                        if is_stderr {
                            stderr_buf.push_str(&line);
                            stderr_buf.push('\n');
                            registry.append_output(task_id, None, Some(&line));
                        } else {
                            stdout_buf.push_str(&line);
                            stdout_buf.push('\n');
                            registry.append_output(task_id, Some(&line), None);
                        }
                    }
                    None => break, // all output drained — process exited
                }
            }
            _ = &mut sleep, if !unlimited => {
                kill_process_group(pid);
                tracing::warn!(command = %command, timeout_ms = timeout_ms, "background: timed out");
                // Drain remaining buffered output
                while let Ok((is_stderr, line)) = line_rx.try_recv() {
                    if is_stderr {
                        registry.append_output(task_id, None, Some(&line));
                    } else {
                        registry.append_output(task_id, Some(&line), None);
                    }
                }
                registry.update_status(task_id, BackgroundTaskStatus::TimedOut);
                guard.done = true;
                return;
            }
            _ = cancel_token.cancelled() => {
                kill_process_group(pid);
                tracing::info!(command = %command, "background: cancelled");
                registry.update_status(task_id, BackgroundTaskStatus::Cancelled);
                guard.done = true;
                return;
            }
        }
    }

    // Reap exit code
    match guard.child.wait().await {
        Ok(status) => {
            guard.done = true;
            registry.finish(task_id, status.code());
        }
        Err(e) => {
            guard.done = true;
            tracing::error!(error = %e, command = %command, "background: wait failed");
            registry.set_error(task_id, format!("wait failed: {}", e));
        }
    }
}

/// Kill the spawned command's process group on timeout/cancel, so children-of-
/// children (e.g. a `java` server launched by `mvn spring-boot:run`) don't leak
/// as orphans once the direct child is gone. `process_group(0)` made the child a
/// group leader (pgid == its pid), so a negative `kill` reaches every descendant.
///
/// The libc syscall is used directly rather than the external `kill` binary:
/// procps' `/usr/bin/kill` on some Linux builds silently fails (returns 0 but
/// signals nothing) for process groups created in the *same session* via
/// `setpgid`, which made the timeout kill a no-op on CI and leaked whole process
/// trees. The raw syscall — the same one the shell builtin uses — has no such
/// argv-parsing surface and works in every environment tested.
#[cfg(unix)]
pub(crate) fn kill_process_group(pgid: Option<u32>) {
    let Some(pgid) = pgid else { return };
    // SAFETY: `pgid` is the pid of the process the tool spawned and is still
    // running (only reached on timeout/cancel before reap). A negative pid
    // targets the whole group. Returns -1 for ESRCH after a successful kill,
    // which is fine.
    unsafe {
        libc::kill(-(pgid as i32), libc::SIGKILL);
    }
}

/// Non-unix fallback: kill only the direct child (no process-group support).
#[cfg(not(unix))]
pub(crate) fn kill_process_group(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    let _ = std::process::Command::new("kill")
        .arg("-9")
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// RAII guard over a spawned child that owns process-group cleanup.
///
/// The engine drives a tool's future with `tokio::select!` and, on cancel, drops
/// it — so a kill inside `call()`'s own `select!` isn't guaranteed to run. This
/// guard runs its group-kill from `Drop` instead, guaranteeing an abandoned
/// command's whole tree is reaped no matter how the future ends (normal, timeout,
/// cancel, or panic). `done` flips to `true` after the child is reaped so a clean
/// completion doesn't kill anything.
struct ProcessGroupGuard {
    pgid: Option<u32>,
    child: tokio::process::Child,
    done: bool,
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if !self.done {
            kill_process_group(self.pgid);
        }
    }
}

#[cfg(test)]
#[path = "local_shell_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "local_shell_proptests.rs"]
mod proptests;

#[cfg(test)]
#[path = "local_shell_prompt_tests.rs"]
mod prompt_tests;
