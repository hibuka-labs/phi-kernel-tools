//! Terminal sessions: the transport seam between a running interactive task
//! and the agent's input/screen access.
//!
//! [`TermSession`] is the trait (read the screen tail, send text/keys, resize,
//! close). [`LocalPty`] is this release's only implementation — a local PTY
//! via `portable-pty`. `SshPty` (remote PTY over the sshr transport) is a
//! reserved slot: same trait, different transport, deliberately not written
//! yet.
//!
//! The screen is a bounded tail buffer ([`ScreenBuffer`]) pumped by a reader
//! thread; interactive prompts arrive without trailing newlines, so the
//! buffer accumulates raw chunks (no line discipline of our own) and exposes
//! a watch handle for "output changed" wakeups.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use portable_pty::{Child, CommandBuilder, MasterPty, PtySize, native_pty_system};
use tokio::sync::watch;

use crate::interaction::policy::key_bytes;

/// Screen buffer cap (bytes). The screen tail is all a prompt match or a
/// model read ever needs; 32K comfortably holds several screens.
const SCREEN_TAIL_LIMIT: usize = 32 * 1024;

/// Bounded screen tail with a change clock and watch handle.
pub struct ScreenBuffer {
    inner: Mutex<ScreenState>,
    /// Bumped on every chunk (and on reader EOF) so waiters wake promptly.
    watch: watch::Sender<u64>,
}

struct ScreenState {
    text: String,
    last_write: Instant,
}

impl ScreenBuffer {
    pub fn new() -> Self {
        let (watch, _) = watch::channel(0u64);
        Self {
            inner: Mutex::new(ScreenState {
                text: String::new(),
                last_write: Instant::now(),
            }),
            watch,
        }
    }

    /// Append a decoded chunk (possibly empty — an empty push still counts as
    /// a change event, which is how reader EOF wakes the supervisor). Bounded
    /// to `SCREEN_TAIL_LIMIT` bytes, sliced at a char boundary.
    pub fn push_chunk(&self, chunk: &str) {
        if !chunk.is_empty() {
            let mut st = self.inner.lock().unwrap();
            st.text.push_str(chunk);
            st.last_write = Instant::now();
            if st.text.len() > SCREEN_TAIL_LIMIT {
                let mut cut = st.text.len() - SCREEN_TAIL_LIMIT;
                while cut < st.text.len() && !st.text.is_char_boundary(cut) {
                    cut += 1;
                }
                st.text.drain(..cut);
            }
        }
        self.watch.send_modify(|seq| *seq += 1);
    }

    /// Last `max_chars` characters of the screen, char-boundary safe.
    pub fn read_tail(&self, max_chars: usize) -> String {
        let st = self.inner.lock().unwrap();
        st.text
            .chars()
            .rev()
            .take(max_chars)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// When output last arrived (the quiet half of prompt detection).
    pub fn last_write(&self) -> Instant {
        self.inner.lock().unwrap().last_write
    }

    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.watch.subscribe()
    }
}

impl Default for ScreenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Transport seam for interactive terminal tasks.
#[async_trait]
pub trait TermSession: Send + Sync {
    /// Screen tail (raw text; strip ANSI at the matcher, not here).
    async fn read_screen(&self, max_chars: usize) -> String;
    /// Send literal text. A trailing newline is the caller's to include.
    async fn send_text(&self, text: &str) -> Result<(), String>;
    /// Send one named key press ([`crate::interaction::policy::KEY_NAMES`] vocabulary).
    async fn send_key(&self, key: &str) -> Result<(), String>;
    async fn resize(&self, cols: u16, rows: u16) -> Result<(), String>;
    async fn close(&self) -> Result<(), String>;
}

/// Local PTY session: `sh -c <command>` with the PTY as its controlling
/// terminal (the child is a session leader, so a process-group kill reaches
/// its whole tree).
pub struct LocalPty {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    master: Mutex<Option<Box<dyn MasterPty + Send>>>,
    child: Mutex<Box<dyn Child + Send + Sync>>,
    screen: Arc<ScreenBuffer>,
    reader_eof: Arc<AtomicBool>,
}

/// Windows ConPTY hands the application's cursor-position query (`ESC[6n`,
/// DSR) to the *host* and waits for the reply on the child's stdin. A real
/// terminal answers it; a raw pipe does not, so the child blocks before it
/// ever prints — MSYS2 bash asks at startup, so every PTY task on Windows
/// hung with an empty screen. Answering it here is what keeps a Windows PTY
/// task alive.
const DSR_QUERY: &[u8] = b"\x1b[6n";

/// The reply a terminal sends for a cursor-position report: row;column, both
/// 1-based. The child only needs a well-formed answer to stop waiting.
const DSR_REPLY: &[u8] = b"\x1b[1;1R";

/// Length of the trailing bytes that could still grow into a DSR query.
///
/// The query is `ESC [ 6 n`; a trailing `ESC`, `ESC [` or `ESC [ 6` may be
/// the start of one, so it is held back until the rest arrives.
fn partial_dsr_len(buf: &[u8]) -> usize {
    for len in [1usize, 2, 3] {
        if buf.len() >= len {
            let start = buf.len() - len;
            if DSR_QUERY.starts_with(&buf[start..]) {
                return len;
            }
        }
    }
    0
}

/// Pull the cursor-position queries out of `pending`.
///
/// Returns the bytes that belong on the screen and whether a query was seen.
/// The tail of a split query is left in `pending` for the next read.
fn extract_dsr(pending: &mut Vec<u8>) -> (Vec<u8>, bool) {
    let mut display = Vec::new();
    let mut query = false;
    loop {
        let found = pending
            .windows(DSR_QUERY.len())
            .position(|w| w == DSR_QUERY);
        let Some(at) = found else { break };
        display.extend_from_slice(&pending[..at]);
        pending.drain(..at + DSR_QUERY.len());
        query = true;
    }
    let hold = partial_dsr_len(pending);
    let keep = pending.len() - hold;
    display.extend_from_slice(&pending[..keep]);
    pending.drain(..keep);
    (display, query)
}

fn io_err<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::other(e.to_string())
}

impl LocalPty {
    /// Open `sh -c <command>` in a fresh PTY of the given size. The reader
    /// thread starts pumping the [`ScreenBuffer`] immediately.
    pub fn open(
        command: &str,
        working_dir: Option<&str>,
        cols: u16,
        rows: u16,
    ) -> std::io::Result<Arc<LocalPty>> {
        let pair = native_pty_system()
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(io_err)?;

        let mut cmd = CommandBuilder::new("sh");
        cmd.arg("-c");
        cmd.arg(command);
        if let Some(dir) = working_dir {
            cmd.cwd(dir);
        }
        let child = pair.slave.spawn_command(cmd).map_err(io_err)?;
        // The child holds its own slave reference; ours can go.
        drop(pair.slave);

        let mut reader = pair.master.try_clone_reader().map_err(io_err)?;
        let writer = pair.master.take_writer().map_err(io_err)?;
        let writer = Arc::new(Mutex::new(writer));

        let screen = Arc::new(ScreenBuffer::new());
        let reader_eof = Arc::new(AtomicBool::new(false));
        {
            let screen = Arc::clone(&screen);
            let reader_eof = Arc::clone(&reader_eof);
            let writer = Arc::clone(&writer);
            std::thread::Builder::new()
                .name("local-pty-reader".to_string())
                .spawn(move || {
                    let mut buf = [0u8; 4096];
                    let mut pending: Vec<u8> = Vec::new();
                    loop {
                        match reader.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                pending.extend_from_slice(&buf[..n]);
                                let (display, saw_query) = extract_dsr(&mut pending);
                                if saw_query {
                                    // Answer the cursor-position query so
                                    // the child stops waiting for it.
                                    let mut w = writer.lock().unwrap();
                                    let _ = w.write_all(DSR_REPLY);
                                    let _ = w.flush();
                                }
                                if !display.is_empty() {
                                    let text = String::from_utf8_lossy(&display);
                                    screen.push_chunk(&text);
                                }
                            }
                        }
                    }
                    reader_eof.store(true, Ordering::SeqCst);
                    // Empty push = a pure change event: EOF wakes waiters.
                    screen.push_chunk("");
                })
                .map_err(io_err)?;
        }

        Ok(Arc::new(LocalPty {
            writer,
            master: Mutex::new(Some(pair.master)),
            child: Mutex::new(child),
            screen,
            reader_eof,
        }))
    }

    pub fn screen(&self) -> Arc<ScreenBuffer> {
        Arc::clone(&self.screen)
    }

    /// Child pid — the PTY session leader, so it doubles as the process-group
    /// id for registry-side kills (shutdown sweep).
    pub fn pid(&self) -> Option<u32> {
        self.child.lock().unwrap().process_id()
    }

    /// The reader has seen EOF (the slave side closed — usually the child
    /// exited). The exit status still needs [`Self::try_wait_exit`].
    pub fn reader_eof(&self) -> bool {
        self.reader_eof.load(Ordering::SeqCst)
    }

    /// Non-blocking exit check; `Some(code)` once the child has exited.
    pub fn try_wait_exit(&self) -> Option<i32> {
        let mut child = self.child.lock().unwrap();
        match child.try_wait() {
            Ok(Some(status)) => Some(status.exit_code() as i32),
            Ok(None) => None,
            Err(_) => Some(-1),
        }
    }

    /// Kill the child's process group (session leader → its whole tree).
    pub fn kill(&self) {
        let pid = self.child.lock().unwrap().process_id();
        crate::local_shell::kill_process_group(pid);
        let _ = self.child.lock().unwrap().kill();
    }
}

#[async_trait]
impl TermSession for LocalPty {
    async fn read_screen(&self, max_chars: usize) -> String {
        self.screen.read_tail(max_chars)
    }

    async fn send_text(&self, text: &str) -> Result<(), String> {
        let mut w = self.writer.lock().unwrap();
        w.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())
    }

    async fn send_key(&self, key: &str) -> Result<(), String> {
        let mut w = self.writer.lock().unwrap();
        w.write_all(&key_bytes(key)).map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())
    }

    async fn resize(&self, cols: u16, rows: u16) -> Result<(), String> {
        let master = self.master.lock().unwrap();
        let Some(master) = master.as_ref() else {
            return Err("session closed".to_string());
        };
        master
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| e.to_string())
    }

    async fn close(&self) -> Result<(), String> {
        self.kill();
        // Dropping the master closes the fd and unblocks the reader thread.
        self.master.lock().unwrap().take();
        Ok(())
    }
}

#[cfg(test)]
#[path = "term_session_tests.rs"]
mod tests;
