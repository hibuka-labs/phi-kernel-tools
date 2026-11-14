//! Tests for [`super`] — LocalPty lifecycle: open/read, input delivery,
//! resize/close, and the screen buffer's bounding.

use super::*;

async fn wait_for_exit(pty: &Arc<LocalPty>) -> Option<i32> {
    for _ in 0..100 {
        if let Some(code) = pty.try_wait_exit() {
            return Some(code);
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    None
}

/// protects: the PTY lifecycle — `open` runs the command, the screen tail
///   shows its output, and the exit code is reaped.
/// fails_when: output never lands on the screen or the exit code is lost.
/// no_claim: prompt detection (the supervisor's job).
#[tokio::test]
async fn open_read_screen_and_exit_code() {
    let pty = LocalPty::open("printf 'hello-pty'", None, 80, 24).unwrap();
    let mut got = String::new();
    for _ in 0..100 {
        got = pty.read_screen(4096).await;
        if got.contains("hello-pty") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        got.contains("hello-pty"),
        "screen never showed output: {got:?}"
    );
    let code = wait_for_exit(&pty).await;
    assert_eq!(code, Some(0), "fixture must exit cleanly");
}

/// protects: input delivery — text plus a named key press reach the child's
///   stdin through the PTY line discipline.
/// fails_when: the bytes are dropped, mangled, or the key map sends wrong bytes.
/// no_claim: what the child does with the input beyond echoing it back.
#[tokio::test]
async fn send_text_and_key_reach_the_child() {
    let pty = LocalPty::open("read line; echo got:$line", None, 80, 24).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    pty.send_text("ping").await.unwrap();
    pty.send_key("enter").await.unwrap();
    let mut got = String::new();
    for _ in 0..100 {
        got = pty.read_screen(4096).await;
        if got.contains("got:ping") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        got.contains("got:ping"),
        "child never echoed the line: {got:?}"
    );
    assert_eq!(wait_for_exit(&pty).await, Some(0));
}

/// protects: resize and close — the size change reaches the kernel and close
///   kills the child's tree (a sleeping fixture must not outlive close).
/// fails_when: close leaks the process or resize errors on a live session.
/// no_claim: downstream apps reacting to SIGWINCH.
#[tokio::test]
async fn resize_and_close_kill_the_tree() {
    let pty = LocalPty::open("sleep 30", None, 80, 24).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    pty.resize(100, 40).await.unwrap();
    assert!(
        pty.try_wait_exit().is_none(),
        "fixture must still be running"
    );
    pty.close().await.unwrap();
    let code = wait_for_exit(&pty).await;
    assert!(code.is_some(), "close must kill the child, not leak it");
}

/// protects: the screen buffer keeps a bounded tail and never slices a
///   multibyte character in half (the CJK underflow lesson).
/// fails_when: the buffer grows past the cap or read_tail cuts mid-character.
/// no_claim: exact byte counts of the retained tail beyond the cap.
#[test]
fn screen_tail_bounds_cjk_safely() {
    let buf = ScreenBuffer::new();
    for _ in 0..2000 {
        buf.push_chunk("日本語のテキスト");
    }
    // Last 10 chars of the repeated 8-char pattern: "スト" of the previous
    // repetition, then a full "日本語のテキスト".
    assert_eq!(buf.read_tail(10), "スト日本語のテキスト");
    // Byte cap holds, and whatever survives starts at a char boundary
    // (read_tail returns well-formed UTF-8 by construction).
    let whole = buf.read_tail(usize::MAX);
    assert!(
        whole.len() <= SCREEN_TAIL_LIMIT,
        "tail grew to {}",
        whole.len()
    );
    assert!(
        whole.len() > SCREEN_TAIL_LIMIT / 2,
        "cap should retain a full tail"
    );
}
