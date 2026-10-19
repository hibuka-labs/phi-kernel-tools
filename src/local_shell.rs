use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::AsyncBufReadExt;
use tokio_util::sync::CancellationToken;

use crate::background_shell::{BackgroundTaskRegistry, BackgroundTaskStatus};

/// Local shell command execution tool.
///
/// Executes arbitrary commands via `sh -c`, with support for timeout,
/// cancellation, and working directory. Optionally supports background
/// execution via an injected `BackgroundTaskRegistry`.
pub struct LocalShellTool {
    timeout_ms: u64,
    registry: Option<Arc<BackgroundTaskRegistry>>,
}

impl LocalShellTool {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            timeout_ms,
            registry: None,
        }
    }

    /// Inject a background task registry. Enables `background: true` support.
    pub fn with_registry(mut self, registry: Arc<BackgroundTaskRegistry>) -> Self {
        self.registry = Some(registry);
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
        Some(0) => match (stdout.is_empty(), stderr.is_empty()) {
            (true, true) => "Command executed successfully with no output.".to_string(),
            (false, true) => stdout.to_string(),
            (true, false) => format!("stderr:\n{}", stderr),
            (false, false) => format!("stdout:\n{}\n\nstderr:\n{}", stdout, stderr),
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

/// Truncate a shell output string to the per-call output budget
/// (`ToolContext::max_output_chars`), keeping both the head and the tail.
///
/// Compile/test errors cluster at the *end* of a command's output, while listing
/// output is meaningful at the *start*, so both ends survive and the elided
/// middle is marked. This mirrors `read_file`/`list_files`: the tool
/// self-truncates so the engine's hard reject never fires for a noisy command.
pub(crate) fn truncate_output(s: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return s.to_string();
    };
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }

    const MARKER: &str = "...[output truncated]";
    // Reserve the marker + two joining newlines; head gets 1/3, tail 2/3.
    let available = max_chars.saturating_sub(MARKER.chars().count() + 2);
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

    format!("{head}\n{MARKER}\n{tail}")
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

        // Stream stdout/stderr line-by-line via reader tasks, so the caller sees
        // live progress instead of a frozen screen for the whole duration. Lines
        // are tagged (stderr vs stdout) and reassembled below.
        let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<(bool, String)>();

        if let Some(stdout) = stdout {
            tokio::spawn(forward_lines(stdout, false, line_tx.clone()));
        }
        if let Some(stderr) = stderr {
            tokio::spawn(forward_lines(stderr, true, line_tx));
        }

        let mut stdout_buf = String::new();
        let mut stderr_buf = String::new();
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
                line = line_rx.recv() => {
                    match line {
                        Some((is_stderr, line)) => {
                            // Live feedback; the final summary is assembled below.
                            ctx.emit_progress(&line);
                            if is_stderr {
                                stderr_buf.push_str(&line);
                                stderr_buf.push('\n');
                            } else {
                                stdout_buf.push_str(&line);
                                stdout_buf.push('\n');
                            }
                        }
                        // Both reader tasks finished → all output drained and the
                        // process exited. Reap the exit code after the loop.
                        None => break,
                    }
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
        let summary = truncate_output(&summary, ctx.max_output_chars);
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
mod tests {
    use super::*;
    use crate::task_registry::TaskStatus;
    use agent_base::tool::content_text;

    #[test]
    fn test_format_result_success() {
        let result = format_result("echo hello", "hello", "", Some(0), false);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_format_result_stderr_only() {
        let result = format_result("cmd", "", "error output", Some(0), false);
        assert_eq!(result, "stderr:\nerror output");
    }

    #[test]
    fn test_format_result_stdout_and_stderr() {
        let result = format_result("cmd", "out", "err", Some(0), false);
        assert_eq!(result, "stdout:\nout\n\nstderr:\nerr");
    }

    #[test]
    fn test_format_result_terminated() {
        let result = format_result("cmd", "", "sigterm", None, false);
        assert!(result.contains("Command Terminated"));
    }

    #[test]
    fn test_format_result_no_output() {
        let result = format_result("true", "", "", Some(0), false);
        assert!(result.contains("no output"));
    }

    #[test]
    fn test_format_result_failure() {
        let result = format_result("false", "", "error", Some(1), false);
        assert!(result.contains("Command Failed"));
        assert!(result.contains("exit code: 1"));
    }

    #[test]
    fn test_format_result_timeout() {
        let result = format_result("sleep 100", "", "", None, true);
        assert!(result.contains("Command Timed Out"));
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
        let v: Value = serde_json::from_str(content_text(&result).trim())
            .expect("background call returns JSON");
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
        assert!(text.trim_end().ends_with("1000"), "tail missing:\n{text}");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn truncate_output_respects_budget(s in ".*", max in 30..200usize) {
            // max must be large enough to fit the marker + head/tail
            let result = truncate_output(&s, Some(max));
            prop_assert!(result.chars().count() <= max,
                "output {} chars exceeds budget {}", result.chars().count(), max);
        }

        #[test]
        fn truncate_output_no_panic(s in ".*", max in 0..500usize) {
            let _ = truncate_output(&s, Some(max));
        }

        #[test]
        fn format_result_does_not_panic(
            command in ".*",
            stdout in ".*",
            stderr in ".*",
            exit_code in proptest::option::of(0i32..128),
            timed_out in proptest::bool::ANY,
        ) {
            let _ = format_result(&command, &stdout, &stderr, exit_code, timed_out);
        }

        #[test]
        fn truncate_output_short_string_unchanged(s in "[a-z]{0,50}") {
            let result = truncate_output(&s, Some(100));
            prop_assert_eq!(result, s);
        }
    }
}
