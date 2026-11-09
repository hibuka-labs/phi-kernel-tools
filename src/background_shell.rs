//! Background shell task registry.
//!
//! Manages background shell tasks spawned by `LocalShellTool` with
//! `background: true`. Provides a shared registry for:
//! - Tracking task status (Running/Done/TimedOut/Cancelled/Error)
//! - Bounded output buffering (head 8K + tail 24K)
//! - Cancellation via `CancellationToken` + process group kill
//! - Async wait via `tokio::sync::watch` channel
//! - GC of completed tasks (driven by `snapshot_all`)
//!
//! This module uses the generic `task_registry` framework internally while
//! exposing shell-specific types and tools.

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::local_shell::{MARKER, kill_process_group, truncate_with_marker};
use crate::spill::{SpillSink, spill_marker};
use crate::task_registry::{TaskEntry, TaskRegistry as GenericRegistry, TaskStatus};

// ── Shell-specific types ──────────────────────────────────────────────

/// Background task status (terminal states are final: once Cancelled/TimedOut/Error is reached, the status never changes).
#[derive(Clone, Debug, PartialEq)]
pub enum BackgroundTaskStatus {
    Running,
    Done,
    TimedOut,
    Cancelled,
    Error(String),
}

impl TaskStatus for BackgroundTaskStatus {
    fn is_terminal(&self) -> bool {
        !matches!(self, BackgroundTaskStatus::Running)
    }

    fn is_wake_worthy(&self) -> bool {
        matches!(
            self,
            BackgroundTaskStatus::Done
                | BackgroundTaskStatus::TimedOut
                | BackgroundTaskStatus::Error(_)
        )
    }
}

/// Shell task entry (business layer implements `TaskEntry`).
#[derive(Debug, Clone)]
pub struct ShellTaskEntry {
    /// Task identifier (bg_* format).
    pub id: String,
    /// Shell command.
    pub command: String,
    /// Working directory.
    pub working_dir: Option<String>,
    /// Per-call timeout fuse in milliseconds — the fact "does this task have
    /// a deadline", registered here because the fuse lives with the task.
    /// `0` = indefinite (daemon-style server/watcher, never auto-killed).
    pub timeout_ms: u64,
    /// Current status.
    status: BackgroundTaskStatus,
    /// Bounded stdout buffer (head 8K + tail 24K).
    stdout: String,
    /// Bounded stderr buffer (head 8K + tail 24K).
    stderr: String,
    /// Process exit code (set on completion).
    exit_code: Option<i32>,
    /// Process-group id of the spawned child (unix only).
    child_pgid: Option<u32>,
}

impl TaskEntry for ShellTaskEntry {
    type Status = BackgroundTaskStatus;

    fn id(&self) -> &str {
        &self.id
    }

    fn status(&self) -> &BackgroundTaskStatus {
        &self.status
    }

    fn set_status(&mut self, status: BackgroundTaskStatus) {
        self.status = status;
    }
}

impl ShellTaskEntry {
    /// Creates a new shell task entry in Running state.
    fn new(
        id: String,
        command: String,
        working_dir: Option<String>,
        pgid: Option<u32>,
        timeout_ms: u64,
    ) -> Self {
        Self {
            id,
            command,
            working_dir,
            timeout_ms,
            status: BackgroundTaskStatus::Running,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            child_pgid: pgid,
        }
    }

    /// Appends output to stdout buffer with bounding.
    pub fn append_stdout(&mut self, line: &str) {
        self.stdout.push_str(line);
        self.stdout.push('\n');
        Self::bound_buffer(&mut self.stdout);
    }

    /// Appends output to stderr buffer with bounding.
    pub fn append_stderr(&mut self, line: &str) {
        self.stderr.push_str(line);
        self.stderr.push('\n');
        Self::bound_buffer(&mut self.stderr);
    }

    /// Sets the exit code (does not change status).
    pub fn set_exit_code(&mut self, exit_code: Option<i32>) {
        self.exit_code = exit_code;
    }

    /// Sets the process group id.
    pub fn set_pgid(&mut self, pgid: Option<u32>) {
        self.child_pgid = pgid;
    }

    /// Returns the process group id.
    pub fn child_pgid(&self) -> Option<u32> {
        self.child_pgid
    }

    /// Bounds a buffer to head 8K + tail 24K bytes, sliced at char boundaries.
    ///
    /// The old version mixed units — byte guard, then `char_count - TAIL_LIMIT`
    /// — which underflowed on multibyte-heavy content (CJK is 3 bytes/char, so
    /// \>32K bytes could still be <24K chars): panic in debug builds, silently
    /// dropped tail in release (2026-10-12 real incident). The panic fired while the
    /// registry lock was held, poisoning it for every later append.
    fn bound_buffer(buf: &mut String) {
        const HEAD_LIMIT: usize = 8 * 1024;
        const TAIL_LIMIT: usize = 24 * 1024;
        const MARKER: &str = "...[truncated]...\n";

        if buf.len() <= HEAD_LIMIT + TAIL_LIMIT {
            return;
        }

        let mut head_end = HEAD_LIMIT.min(buf.len());
        while head_end > 0 && !buf.is_char_boundary(head_end) {
            head_end -= 1;
        }
        let mut tail_start = buf.len().saturating_sub(TAIL_LIMIT);
        while tail_start < buf.len() && !buf.is_char_boundary(tail_start) {
            tail_start += 1;
        }
        let bounded = format!("{}{MARKER}{}", &buf[..head_end], &buf[tail_start..]);
        *buf = bounded;
    }
}

// ── Backward-compatible types ─────────────────────────────────────────

/// Background task snapshot (for queries; bounded output buffers).
///
/// This provides backward compatibility with the TUI layer.
#[derive(Clone, Debug)]
pub struct BackgroundTaskSnapshot {
    pub id: String,
    pub command: String,
    pub working_dir: Option<String>,
    /// Per-call timeout fuse in ms; `0` = indefinite (daemon-style).
    pub timeout_ms: u64,
    pub status: BackgroundTaskStatus,
    pub started_at: Instant,
    pub finished_at: Option<Instant>,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    /// Whether `task_output` has already returned this task's terminal output
    /// (consumed). Consumed entries are GC-eligible immediately.
    pub consumed: bool,
}

/// The last `max_bytes` bytes of `s`, never cutting a UTF-8 code point.
///
/// Returns a suffix of `s` of at most `max_bytes` bytes (measured in the
/// UTF-8 encoding) aligned to a char boundary. Empty when `s` is empty or
/// `max_bytes` is zero.
pub fn tail_utf8(s: &str, max_bytes: usize) -> &str {
    if max_bytes == 0 || s.is_empty() {
        return "";
    }
    if s.len() <= max_bytes {
        return s;
    }
    let mut start = s.len() - max_bytes;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

impl BackgroundTaskSnapshot {
    /// Creates a snapshot from a generic registry snapshot.
    fn from_generic(snap: &crate::task_registry::Snapshot<ShellTaskEntry>) -> Self {
        Self {
            id: snap.id.clone(),
            command: snap.entry.command.clone(),
            working_dir: snap.entry.working_dir.clone(),
            timeout_ms: snap.entry.timeout_ms,
            status: snap.entry.status().clone(),
            started_at: snap.created_at,
            finished_at: snap.finished_at,
            stdout: snap.entry.stdout.clone(),
            stderr: snap.entry.stderr.clone(),
            exit_code: snap.entry.exit_code,
            consumed: snap.consumed,
        }
    }

    /// The tail of this task's combined output (stderr after stdout), capped
    /// to `max_bytes` UTF-8-safe. Used to build wake reports that survive
    /// registry GC.
    pub fn output_tail(&self, max_bytes: usize) -> String {
        let mut combined = self.stdout.clone();
        if !self.stderr.is_empty() {
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str("[stderr]\n");
            combined.push_str(&self.stderr);
        }
        tail_utf8(&combined, max_bytes).to_string()
    }
}

// ── Registry wrapper ──────────────────────────────────────────────────

/// Shared background task registry.
///
/// Wraps the generic `TaskRegistry` with shell-specific convenience methods.
/// Constructed via `new(max_tasks)` and shared as `Arc<BackgroundTaskRegistry>`.
/// The registry is injected into `LocalShellTool`, `TaskOutputTool`, and
/// `TaskCancelTool` at build time. The TUI also holds a clone for tick-based
/// polling via `snapshot_all`.
pub struct BackgroundTaskRegistry {
    inner: Arc<GenericRegistry<ShellTaskEntry>>,
}

impl std::fmt::Debug for BackgroundTaskRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundTaskRegistry")
            .field("inner", &"<GenericRegistry>")
            .finish()
    }
}

impl BackgroundTaskRegistry {
    /// Create a new registry with the given maximum concurrent task limit.
    pub fn new(max_tasks: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: GenericRegistry::<ShellTaskEntry>::new(max_tasks),
        })
    }

    /// Register a new background task.
    ///
    /// `timeout_ms` is the kill-fuse the executor will arm (`0` = indefinite,
    /// daemon-style); it is recorded on the entry so the UI can tell "waiting
    /// on a bounded job" apart from "a server is just running".
    ///
    /// Returns `Ok(task_id)` (format `bg_{short_uuid}`) or
    /// `Err("too many background tasks")` if the limit is reached.
    pub fn register(
        &self,
        command: &str,
        working_dir: Option<&str>,
        cancel_token: CancellationToken,
        pgid: Option<u32>,
        timeout_ms: u64,
    ) -> Result<String, String> {
        let id = format!("bg_{}", &Uuid::new_v4().simple().to_string()[..8]);
        let entry = ShellTaskEntry::new(
            id.clone(),
            command.to_string(),
            working_dir.map(String::from),
            pgid,
            timeout_ms,
        );

        self.inner
            .register(entry, cancel_token)
            .map_err(|_| "too many background tasks".to_string())?;

        Ok(id)
    }

    /// Append output lines from the background executor.
    ///
    /// Maintains a bounded buffer per stream: head 8K + tail 24K chars.
    pub fn append_output(&self, id: &str, stdout_line: Option<&str>, stderr_line: Option<&str>) {
        self.inner.with_entry_mut(id, |entry| {
            if let Some(line) = stdout_line {
                entry.append_stdout(line);
            }
            if let Some(line) = stderr_line {
                entry.append_stderr(line);
            }
        });
    }

    /// Set the process-group id for a task.
    pub fn set_pgid(&self, id: &str, pgid: Option<u32>) {
        self.inner.with_entry_mut(id, |entry| {
            entry.set_pgid(pgid);
        });
    }

    /// Set the exit code, mark as `Done`, and record `finished_at`.
    ///
    /// **Terminal-state protection**: if the task is already in a terminal
    /// state, this call is a no-op.
    pub fn finish(&self, id: &str, exit_code: Option<i32>) {
        // Set exit code first
        self.inner.with_entry_mut(id, |entry| {
            entry.set_exit_code(exit_code);
        });
        // Then update status (framework handles terminal-state protection and finished_at)
        self.inner.update_status(id, BackgroundTaskStatus::Done);
    }

    /// Update the task status.
    ///
    /// **Terminal-state protection**: if the task is already in a terminal
    /// state, the new status is silently ignored.
    pub fn update_status(&self, id: &str, status: BackgroundTaskStatus) {
        self.inner.update_status(id, status);
    }

    /// Set an error status. Respects terminal-state protection.
    pub fn set_error(&self, id: &str, error: String) {
        self.update_status(id, BackgroundTaskStatus::Error(error));
    }

    /// Get a snapshot of a single task.
    pub fn snapshot(&self, id: &str) -> Option<BackgroundTaskSnapshot> {
        self.inner
            .snapshot(id)
            .map(|snap| BackgroundTaskSnapshot::from_generic(&snap))
    }

    /// Get snapshots of all tasks. Pure read — pair with [`Self::gc`] so a
    /// terminal state is always observed before it can be swept away.
    pub fn snapshot_all(&self) -> Vec<BackgroundTaskSnapshot> {
        self.inner
            .snapshot_all()
            .iter()
            .map(BackgroundTaskSnapshot::from_generic)
            .collect()
    }

    /// GC sweep: drop consumed entries, or unconsumed ones finished longer
    /// than `gc_ttl` ago. Running tasks are never dropped.
    pub fn gc(&self, gc_ttl: Duration) {
        self.inner.gc(gc_ttl);
    }

    /// Mark a task's output as consumed (successful terminal retrieval).
    /// Idempotent; no-op on running tasks. See `TaskRegistry::mark_consumed`.
    pub fn mark_consumed(&self, id: &str) -> bool {
        self.inner.mark_consumed(id)
    }

    /// Count of running (non-terminal) tasks.
    pub fn running_count(&self) -> usize {
        self.inner.running_count()
    }

    /// Cancel a background task.
    ///
    /// Triggers the `CancellationToken`, sets status to `Cancelled`.
    /// Returns `true` if the task was found and cancelled, `false` otherwise.
    pub fn cancel(&self, id: &str) -> bool {
        let cancelled = self.inner.cancel(id);
        if cancelled {
            // Set the business-layer status to Cancelled
            self.inner
                .update_status(id, BackgroundTaskStatus::Cancelled);
        }
        cancelled
    }

    /// Wait for a task to reach a terminal state.
    ///
    /// Returns `Some(snapshot)` when the task finishes, or `None` on timeout
    /// or if the task doesn't exist.
    pub async fn wait(&self, id: &str, timeout: Duration) -> Option<BackgroundTaskSnapshot> {
        self.inner
            .wait(id, timeout)
            .await
            .map(|snap| BackgroundTaskSnapshot::from_generic(&snap))
    }

    /// Returns a `watch::Receiver` for TUI to detect status changes.
    pub fn watch(&self) -> tokio::sync::watch::Receiver<()> {
        self.inner.watch()
    }

    /// Explicit shutdown: cancel all running tasks, SIGKILL all process groups,
    /// and clear the registry.
    ///
    /// Must be called at session end. Does NOT rely on `Drop` because
    /// executor tasks hold `Arc<Self>`.
    pub fn shutdown(&self) {
        // Kill process groups for running tasks before shutting down
        // We need to collect pgids first to avoid holding the lock during kill_process_group
        let pgids: Vec<Option<u32>> = {
            let map = self.inner.slots.read().unwrap();
            map.values()
                .filter(|slot| !slot.entry.status().is_terminal())
                .map(|slot| slot.entry.child_pgid())
                .collect()
        };

        for pgid in pgids {
            kill_process_group(pgid);
        }

        self.inner.shutdown();
    }
}

// ── Tools ─────────────────────────────────────────────────────────────

/// `task_output` — query or wait for a background task's result.
#[derive(Clone)]
pub struct TaskOutputTool {
    registry: Arc<BackgroundTaskRegistry>,
    spill: Option<Arc<dyn SpillSink>>,
}

impl TaskOutputTool {
    pub fn new(registry: Arc<BackgroundTaskRegistry>) -> Self {
        Self {
            registry,
            spill: None,
        }
    }

    /// Inject a spill sink for truncated status output — see
    /// `LocalShellTool::with_spill` for the honesty rule.
    pub fn with_spill(mut self, spill: Arc<dyn SpillSink>) -> Self {
        self.spill = Some(spill);
        self
    }
}

fn format_status(snap: &BackgroundTaskSnapshot, max_chars: Option<usize>, marker: &str) -> Value {
    let elapsed_ms = snap.started_at.elapsed().as_millis();
    // Two payload fields must fit the engine's output budget together with the
    // JSON envelope — self-truncate like `execute_command`, or the pipeline
    // hard-rejects the whole result (session 20261018_7844215b: `task_output`
    // died as `ToolOutputTooLarge` on a fat fan-in notice).
    let field_budget = max_chars.map(|m| (m / 2).saturating_sub(256));
    let trim = |s: &str| -> String { truncate_with_marker(s, field_budget, marker) };
    let stdout = trim(&snap.stdout);
    let stderr = trim(&snap.stderr);
    match &snap.status {
        BackgroundTaskStatus::Running => json!({
            "status": "running",
            "elapsed_ms": elapsed_ms,
            "partial_stdout": stdout,
            "partial_stderr": stderr,
        }),
        BackgroundTaskStatus::Done => json!({
            "status": "done",
            "exit_code": snap.exit_code,
            "elapsed_ms": elapsed_ms,
            "stdout": stdout,
            "stderr": stderr,
        }),
        BackgroundTaskStatus::TimedOut => json!({
            "status": "timed_out",
            "exit_code": snap.exit_code,
            "elapsed_ms": elapsed_ms,
            "stdout": stdout,
            "stderr": stderr,
        }),
        BackgroundTaskStatus::Cancelled => json!({
            "status": "cancelled",
            "exit_code": snap.exit_code,
            "elapsed_ms": elapsed_ms,
            "stdout": stdout,
            "stderr": stderr,
        }),
        BackgroundTaskStatus::Error(msg) => json!({
            "status": "error",
            "error": msg,
            "elapsed_ms": elapsed_ms,
            "stdout": stdout,
            "stderr": stderr,
        }),
    }
}

/// Render a task status envelope within `max_chars`.
///
/// Truncation can fire twice — per payload field (`format_status`) and on the
/// serialized envelope (JSON escaping can inflate past the budget) — and
/// either way the middle is lost. When any cut fires, the complete
/// pre-truncation JSON is spilled (kind `"shell_output"`) and both marker
/// layers name the artifact; without a sink, or on a spill failure, today's
/// plain-marker output stands byte-for-byte.
fn render_status(
    snap: &BackgroundTaskSnapshot,
    max_chars: Option<usize>,
    spill: Option<&dyn SpillSink>,
) -> String {
    let render = |marker: &str| -> String {
        let text = serde_json::to_string_pretty(&format_status(snap, max_chars, marker))
            .unwrap_or_else(|_| "{}".to_string());
        truncate_with_marker(&text, max_chars, marker)
    };
    let plain = render(MARKER);
    // Full untrimmed JSON — the honest spill body. Equal to `plain` iff no cut
    // fired at either layer.
    let full_text = serde_json::to_string_pretty(&format_status(snap, None, MARKER))
        .unwrap_or_else(|_| "{}".to_string());
    if plain == full_text {
        return plain;
    }
    let Some(marker) = spill_marker(spill, "shell_output", &full_text) else {
        return plain;
    };
    render(&marker)
}

#[async_trait]
impl Tool for TaskOutputTool {
    fn name(&self) -> &'static str {
        "task_output"
    }

    fn description(&self) -> &'static str {
        "Check or wait for a background SHELL task's result. Use `wait: true` to block until completion.\n\
         Scope guard (session 20260919_f65b754c: 12 wasted polls): this only\n\
         accepts `bg_*` ids returned by `execute_command` with `background: true`.\n\
         Sub-agent results are NOT fetchable here — they are pushed to you\n\
         automatically when the agent finishes; end your turn to receive them.\n\
         NEVER invent ids like `bg_<agent-name>` for a spawned agent."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Background task ID from execute_command (bg_* format). Returns error if task was GC'd (consumed via an earlier call, or ~30 min after completion)."
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, blocks until the task completes. If false, returns current state immediately.",
                    "default": false
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Max wait time in ms when wait=true. Default: 120000 (2 min). Range: 1000-300000.",
                    "default": 120000
                }
            },
            "required": ["task_id"]
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: self.description().to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    fn timeout_ms(&self) -> Option<u64> {
        // Must exceed the internal wait timeout (max 300s) so the tool's own
        // timeout fires before the engine's hard-reject `[Tool Timeout]`.
        Some(310_000)
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let task_id = args
            .get("task_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        if task_id.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: No task_id provided.".to_string(),
            )]);
        }

        let should_wait = args.get("wait").and_then(Value::as_bool).unwrap_or(false);

        if should_wait {
            let timeout_ms = args
                .get("timeout_ms")
                .and_then(Value::as_u64)
                .unwrap_or(120_000)
                .clamp(1_000, 300_000);
            let timeout = Duration::from_millis(timeout_ms);

            match self.registry.wait(&task_id, timeout).await {
                Some(snap) => {
                    // Full terminal output delivered — the entry becomes
                    // GC-eligible. Partial (still-running) reads don't count.
                    if snap.status.is_terminal() {
                        self.registry.mark_consumed(&task_id);
                    }
                    Ok(vec![Content::text(render_status(
                        &snap,
                        ctx.max_output_chars,
                        self.spill.as_deref(),
                    ))])
                }
                None => Ok(vec![Content::text(format!(
                    "{}",
                    json!({
                        "status": "not_found",
                        "message": not_found_message(&task_id)
                    })
                ))]),
            }
        } else {
            match self.registry.snapshot(&task_id) {
                Some(snap) => {
                    if snap.status.is_terminal() {
                        self.registry.mark_consumed(&task_id);
                    }
                    Ok(vec![Content::text(render_status(
                        &snap,
                        ctx.max_output_chars,
                        self.spill.as_deref(),
                    ))])
                }
                None => Ok(vec![Content::text(format!(
                    "{}",
                    json!({
                        "status": "not_found",
                        "message": not_found_message(&task_id)
                    })
                ))]),
            }
        }
    }
}

/// not_found fact + the one redirect that matters. Session
/// 20260919_f65b754c: a parent polled `bg_analyze-*` (ids fabricated from
/// sub-agent names) 12 times while its 4 children ran — each a wasted call.
/// The error is the teachable moment: this registry holds background SHELL
/// tasks only, and a sub-agent's report is pushed, never fetched.
fn not_found_message(task_id: &str) -> String {
    format!(
        "Task {} not found. Its output was already fetched via an earlier task_output call, \n         or the entry was cleaned up ~30 minutes after completion. \
         This registry holds background SHELL tasks only — a spawned sub-agent's \
         report is pushed to you automatically; end your turn to receive it.",
        task_id
    )
}

/// `task_cancel` — terminate a running background task.
#[derive(Clone)]
pub struct TaskCancelTool {
    registry: Arc<BackgroundTaskRegistry>,
}

impl TaskCancelTool {
    pub fn new(registry: Arc<BackgroundTaskRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for TaskCancelTool {
    fn name(&self) -> &'static str {
        "task_cancel"
    }

    fn description(&self) -> &'static str {
        "Terminate a running background task by task_id."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Background task ID to cancel (bg_* format)"
                }
            },
            "required": ["task_id"]
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: self.description().to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let task_id = args
            .get("task_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        if task_id.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: No task_id provided.".to_string(),
            )]);
        }

        // Check existence and status before cancelling.
        match self.registry.snapshot(&task_id) {
            None => Ok(vec![Content::text(format!("Task {} not found.", task_id))]),
            Some(snap) if snap.status.is_terminal() => Ok(vec![Content::text(format!(
                "Task {} already finished with status: {:?}.",
                task_id, snap.status
            ))]),
            Some(_) => {
                // cancel() returns true only for non-terminal tasks.
                self.registry.cancel(&task_id);
                Ok(vec![Content::text(format!("Task {} cancelled.", task_id))])
            }
        }
    }
}

#[cfg(test)]
#[path = "background_shell_tests.rs"]
mod tests;
