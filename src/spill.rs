//! Injection seam for shell-output overflow.
//!
//! When a shell tool result must be truncated, the full pre-truncation body is
//! worth keeping retrievable (`read_file` has `offset` continuation; shell
//! output has none). Where that body goes is a host concern: `codexec`'s
//! artifact store fits, but this crate cannot depend on it (dep direction), so
//! the concrete store is injected through [`crate::spill::SpillSink`] — same
//! construction pattern as `with_registry`.
//!
//! Honesty rule: a missing sink or a failed write keeps today's plain
//! truncation marker. Never a fake path, never silent.

use std::path::PathBuf;

#[cfg(all(test, feature = "shell"))]
use std::sync::Arc;

/// Sink for full copies of shell output cut from a tool result.
pub trait SpillSink: Send + Sync {
    /// Persist `body` under `kind` (shell tools use `"shell_output"`) and
    /// return the path the model can read back. `Err` = no artifact written;
    /// callers must fall back to the plain truncation marker.
    fn spill(&self, kind: &str, body: &str) -> Result<PathBuf, String>;
}

/// Marker text pointing at a spilled full copy of a truncated body.
///
/// Shape (wire string, English): `...[output truncated; full output: {path}
/// (use read_file with offset/limit)]`. Returns `None` when there is no sink or
/// the write failed — the caller then keeps the plain
/// `...[output truncated]` marker.
#[cfg(feature = "shell")]
pub(crate) fn spill_marker(sink: Option<&dyn SpillSink>, kind: &str, body: &str) -> Option<String> {
    let path = sink?.spill(kind, body).ok()?;
    Some(format!(
        "...[output truncated; full output: {} (use read_file with offset/limit)]",
        path.display()
    ))
}

// ── Test support ──────────────────────────────────────────────────────

/// One recorded spill.
#[cfg(all(test, feature = "shell"))]
#[derive(Clone)]
pub(crate) struct SpillRecord {
    pub(crate) kind: String,
    pub(crate) body: String,
    pub(crate) path: PathBuf,
}

/// Test double for [`SpillSink`]: writes each body into a temp dir and records
/// the call. `new_failing` simulates a write failure (the degradation path).
#[cfg(all(test, feature = "shell"))]
pub(crate) struct StubSink {
    dir: PathBuf,
    fail: bool,
    calls: std::sync::Mutex<Vec<SpillRecord>>,
    seq: std::sync::atomic::AtomicUsize,
}

#[cfg(all(test, feature = "shell"))]
impl StubSink {
    pub(crate) fn new(dir: &std::path::Path) -> Arc<Self> {
        Arc::new(Self {
            dir: dir.to_path_buf(),
            fail: false,
            calls: std::sync::Mutex::new(Vec::new()),
            seq: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub(crate) fn new_failing(dir: &std::path::Path) -> Arc<Self> {
        Arc::new(Self {
            dir: dir.to_path_buf(),
            fail: true,
            calls: std::sync::Mutex::new(Vec::new()),
            seq: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub(crate) fn calls(&self) -> Vec<SpillRecord> {
        self.calls.lock().unwrap().clone()
    }
}

#[cfg(all(test, feature = "shell"))]
impl SpillSink for StubSink {
    fn spill(&self, kind: &str, body: &str) -> Result<PathBuf, String> {
        let n = self.seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.fail {
            return Err("stub sink: write failed".to_string());
        }
        let path = self.dir.join(format!("spill_{n}"));
        std::fs::write(&path, body).map_err(|e| e.to_string())?;
        self.calls.lock().unwrap().push(SpillRecord {
            kind: kind.to_string(),
            body: body.to_string(),
            path: path.clone(),
        });
        Ok(path)
    }
}
