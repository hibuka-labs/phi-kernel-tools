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

use crate::local_shell::kill_process_group;
use crate::task_registry::{TaskStatus, TaskEntry, TaskRegistry as GenericRegistry};

// ── Shell-specific types ──────────────────────────────────────────────

/// 后台任务状态（终态不可逆：一旦进入 Cancelled/TimedOut/Error，不再变化）
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

/// Shell 任务条目（业务层实现 TaskEntry）
#[derive(Debug, Clone)]
pub struct ShellTaskEntry {
    /// Task identifier (bg_* format).
    pub id: String,
    /// Shell command.
    pub command: String,
    /// Working directory.
    pub working_dir: Option<String>,
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
    fn new(id: String, command: String, working_dir: Option<String>, pgid: Option<u32>) -> Self {
        Self {
            id,
            command,
            working_dir,
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

    /// Bounds a buffer to head 8K + tail 24K chars.
    fn bound_buffer(buf: &mut String) {
        const HEAD_LIMIT: usize = 8 * 1024;
        const TAIL_LIMIT: usize = 24 * 1024;
        const MARKER: &str = "...[truncated]...\n";

        if buf.len() <= HEAD_LIMIT + TAIL_LIMIT {
            return;
        }

        let head: String = buf.chars().take(HEAD_LIMIT).collect();
        let tail: String = {
            let char_count = buf.chars().count();
            buf.chars()
                .skip(char_count - TAIL_LIMIT)
                .collect()
        };
        *buf = format!("{head}{MARKER}{tail}");
    }
}

// ── Backward-compatible types ─────────────────────────────────────────

/// 后台任务快照（用于查询，bounded 输出缓冲）
///
/// This provides backward compatibility with the TUI layer.
#[derive(Clone, Debug)]
pub struct BackgroundTaskSnapshot {
    pub id: String,
    pub command: String,
    pub working_dir: Option<String>,
    pub status: BackgroundTaskStatus,
    pub started_at: Instant,
    pub finished_at: Option<Instant>,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

impl BackgroundTaskSnapshot {
    /// Creates a snapshot from a generic registry snapshot.
    fn from_generic(snap: &crate::task_registry::Snapshot<ShellTaskEntry>) -> Self {
        Self {
            id: snap.id.clone(),
            command: snap.entry.command.clone(),
            working_dir: snap.entry.working_dir.clone(),
            status: snap.entry.status().clone(),
            started_at: snap.created_at,
            finished_at: snap.finished_at,
            stdout: snap.entry.stdout.clone(),
            stderr: snap.entry.stderr.clone(),
            exit_code: snap.entry.exit_code,
        }
    }
}

// ── Registry wrapper ──────────────────────────────────────────────────

/// 共享后台任务注册表。
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
    /// Returns `Ok(task_id)` (format `bg_{short_uuid}`) or
    /// `Err("too many background tasks")` if the limit is reached.
    pub fn register(
        &self,
        command: &str,
        working_dir: Option<&str>,
        cancel_token: CancellationToken,
        pgid: Option<u32>,
    ) -> Result<String, String> {
        let id = format!("bg_{}", &Uuid::new_v4().simple().to_string()[..8]);
        let entry = ShellTaskEntry::new(
            id.clone(),
            command.to_string(),
            working_dir.map(String::from),
            pgid,
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

    /// Get snapshots of all tasks, and GC entries finished longer than `gc_ttl`.
    ///
    /// Designed to be called from the TUI tick loop (every 250ms).
    pub fn snapshot_all(&self, gc_ttl: Duration) -> Vec<BackgroundTaskSnapshot> {
        self.inner
            .snapshot_all(gc_ttl)
            .iter()
            .map(|snap| BackgroundTaskSnapshot::from_generic(snap))
            .collect()
    }

    /// Cancel a background task.
    ///
    /// Triggers the `CancellationToken`, sets status to `Cancelled`.
    /// Returns `true` if the task was found and cancelled, `false` otherwise.
    pub fn cancel(&self, id: &str) -> bool {
        let cancelled = self.inner.cancel(id);
        if cancelled {
            // Set the business-layer status to Cancelled
            self.inner.update_status(id, BackgroundTaskStatus::Cancelled);
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
}

impl TaskOutputTool {
    pub fn new(registry: Arc<BackgroundTaskRegistry>) -> Self {
        Self { registry }
    }
}

fn format_status(snap: &BackgroundTaskSnapshot) -> Value {
    let elapsed_ms = snap.started_at.elapsed().as_millis();
    match &snap.status {
        BackgroundTaskStatus::Running => json!({
            "status": "running",
            "elapsed_ms": elapsed_ms,
            "partial_stdout": snap.stdout,
            "partial_stderr": snap.stderr,
        }),
        BackgroundTaskStatus::Done => json!({
            "status": "done",
            "exit_code": snap.exit_code,
            "elapsed_ms": elapsed_ms,
            "stdout": snap.stdout,
            "stderr": snap.stderr,
        }),
        BackgroundTaskStatus::TimedOut => json!({
            "status": "timed_out",
            "exit_code": snap.exit_code,
            "elapsed_ms": elapsed_ms,
            "stdout": snap.stdout,
            "stderr": snap.stderr,
        }),
        BackgroundTaskStatus::Cancelled => json!({
            "status": "cancelled",
            "exit_code": snap.exit_code,
            "elapsed_ms": elapsed_ms,
            "stdout": snap.stdout,
            "stderr": snap.stderr,
        }),
        BackgroundTaskStatus::Error(msg) => json!({
            "status": "error",
            "error": msg,
            "elapsed_ms": elapsed_ms,
            "stdout": snap.stdout,
            "stderr": snap.stderr,
        }),
    }
}

#[async_trait]
impl Tool for TaskOutputTool {
    fn name(&self) -> &'static str {
        "task_output"
    }

    fn description(&self) -> &'static str {
        "Check or wait for a background task's result. Use `wait: true` to block until completion."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Background task ID from execute_command (bg_* format). Returns error if task was GC'd (5 min TTL)."
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

        let should_wait = args
            .get("wait")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        if should_wait {
            let timeout_ms = args
                .get("timeout_ms")
                .and_then(Value::as_u64)
                .unwrap_or(120_000)
                .clamp(1_000, 300_000);
            let timeout = Duration::from_millis(timeout_ms);

            match self.registry.wait(&task_id, timeout).await {
                Some(snap) => Ok(vec![Content::text(
                    serde_json::to_string_pretty(&format_status(&snap))
                        .unwrap_or_else(|_| "{}".to_string()),
                )]),
                None => Ok(vec![Content::text(format!(
                    "{}",
                    json!({
                        "status": "not_found",
                        "message": format!(
                            "Task {} not found. It may have been cleaned up after 5 minutes of completion.",
                            task_id
                        )
                    })
                ))]),
            }
        } else {
            match self.registry.snapshot(&task_id) {
                Some(snap) => Ok(vec![Content::text(
                    serde_json::to_string_pretty(&format_status(&snap))
                        .unwrap_or_else(|_| "{}".to_string()),
                )]),
                None => Ok(vec![Content::text(format!(
                    "{}",
                    json!({
                        "status": "not_found",
                        "message": format!(
                            "Task {} not found. It may have been cleaned up after 5 minutes of completion.",
                            task_id
                        )
                    })
                ))]),
            }
        }
    }
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
            None => Ok(vec![Content::text(format!(
                "Task {} not found.",
                task_id
            ))]),
            Some(snap) if snap.status.is_terminal() => Ok(vec![Content::text(format!(
                "Task {} already finished with status: {:?}.",
                task_id, snap.status
            ))]),
            Some(_) => {
                // cancel() returns true only for non-terminal tasks.
                self.registry.cancel(&task_id);
                Ok(vec![Content::text(format!(
                    "Task {} cancelled.",
                    task_id
                ))])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助：构造一个 registry + cancel token。
    fn setup(max_tasks: usize) -> (Arc<BackgroundTaskRegistry>, CancellationToken) {
        let reg = BackgroundTaskRegistry::new(max_tasks);
        let token = CancellationToken::new();
        (reg, token)
    }

    // ── register ──

    #[test]
    fn register_returns_bg_id() {
        let (reg, token) = setup(4);
        let id = reg.register("echo hi", None, token, None).unwrap();
        assert!(id.starts_with("bg_"), "id should start with bg_: {id}");
        assert_eq!(id.len(), 11, "bg_ + 8 hex chars = 11");
    }

    #[test]
    fn register_records_command_and_dir() {
        let (reg, token) = setup(4);
        let id = reg.register("ls -la", Some("/tmp"), token, None).unwrap();
        let snap = reg.snapshot(&id).unwrap();
        assert_eq!(snap.command, "ls -la");
        assert_eq!(snap.working_dir.as_deref(), Some("/tmp"));
        assert_eq!(snap.status, BackgroundTaskStatus::Running);
    }

    #[test]
    fn register_respects_max_tasks() {
        let (reg, _token) = setup(2);
        let t1 = CancellationToken::new();
        let t2 = CancellationToken::new();
        let t3 = CancellationToken::new();
        reg.register("cmd1", None, t1, None).unwrap();
        reg.register("cmd2", None, t2, None).unwrap();
        let err = reg.register("cmd3", None, t3, None).unwrap_err();
        assert!(err.contains("too many"));
    }

    #[test]
    fn register_allows_new_after_terminal() {
        let (reg, _) = setup(2);
        let t1 = CancellationToken::new();
        let t2 = CancellationToken::new();
        let t3 = CancellationToken::new();
        let id1 = reg.register("cmd1", None, t1, None).unwrap();
        reg.register("cmd2", None, t2, None).unwrap();
        reg.finish(&id1, Some(0));
        reg.register("cmd3", None, t3, None).unwrap();
    }

    // ── finish / update_status ──

    #[test]
    fn finish_sets_done_and_exit_code() {
        let (reg, token) = setup(4);
        let id = reg.register("cmd", None, token, None).unwrap();
        reg.finish(&id, Some(42));
        let snap = reg.snapshot(&id).unwrap();
        assert_eq!(snap.status, BackgroundTaskStatus::Done);
        assert_eq!(snap.exit_code, Some(42));
        assert!(snap.finished_at.is_some());
    }

    #[test]
    fn terminal_state_protection() {
        let (reg, token) = setup(4);
        let id = reg.register("cmd", None, token, None).unwrap();
        reg.finish(&id, Some(0));
        // Try to update to Error — should be ignored.
        reg.set_error(&id, "boom".into());
        let snap = reg.snapshot(&id).unwrap();
        assert_eq!(snap.status, BackgroundTaskStatus::Done);
    }

    // ── snapshot_all / GC ──

    #[test]
    fn snapshot_all_gc_removes_old_entries() {
        let (reg, token) = setup(4);
        let id = reg.register("cmd", None, token, None).unwrap();
        reg.finish(&id, Some(0));

        // With long TTL, should still be present.
        let snaps = reg.snapshot_all(Duration::from_secs(60));
        assert_eq!(snaps.len(), 1);

        // With zero TTL, should be GC'd.
        let snaps = reg.snapshot_all(Duration::ZERO);
        assert_eq!(snaps.len(), 0);
    }
}
