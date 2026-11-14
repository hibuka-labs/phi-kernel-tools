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
    writer: Mutex<Box<dyn Write + Send>>,
    master: Mutex<Option<Box<dyn MasterPty + Send>>>,
    child: Mutex<Box<dyn Child + Send + Sync>>,
    screen: Arc<ScreenBuffer>,
    reader_eof: Arc<AtomicBool>,
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

        let screen = Arc::new(ScreenBuffer::new());
        let reader_eof = Arc::new(AtomicBool::new(false));
        {
            let screen = Arc::clone(&screen);
            let reader_eof = Arc::clone(&reader_eof);
            std::thread::Builder::new()
                .name("local-pty-reader".to_string())
                .spawn(move || {
                    let mut buf = [0u8; 4096];
                    loop {
                        match reader.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                let text = String::from_utf8_lossy(&buf[..n]);
                                screen.push_chunk(&text);
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
            writer: Mutex::new(writer),
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
