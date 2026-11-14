//! Interactive-prompt detection: a pure, policy-free matcher over terminal
//! output tails.
//!
//! The model only ever sees tool results, so "the command is stuck on a
//! Y/N prompt" must materialize *as* a tool result. This module answers one
//! question — does this output tail look like a prompt waiting for input, and
//! what shape is it — and nothing else. It never decides what to send:
//! [`PromptClass`] reports `kind` + `default_side` + `suggested` (the
//! deterministic safe answer for the class, or `None` for classes that must
//! never be auto-answered), and the answer policy lives in the product layer.
//!
//! Consumers:
//! - `local_shell` foreground: early-kill when the tail matches **and** output
//!   has been quiet for [`PROMPT_QUIET_MS`], plus a tail annotation on timeout.
//! - PTY task surface (P1): watch screen tails, same vocabulary.

mod patterns;

pub mod policy;

use patterns::ROWS;

pub use policy::{PromptPolicy, SendInput};

/// How long the output must stay quiet after a tail match before the process
/// is treated as blocked on a prompt (foreground early-kill dual condition).
/// A log line that merely *mentions* `[Y/n]` while the run keeps producing
/// output never satisfies this, so healthy runs are never killed.
pub const PROMPT_QUIET_MS: u64 = 3_000;

/// How much of the output tail (chars) is scanned for prompts and echoed back
/// in the `[waiting for input]` block.
pub const TAIL_CHARS: usize = 300;

/// Shape of an interactive prompt found on an output tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptClass {
    pub kind: PromptKind,
    /// The highlighted default of a confirm prompt, if any (`[Y/n]` → `Y`,
    /// `[y/N]` → `N`). Reports shape only — the default is never sent here.
    pub default_side: Option<char>,
    pub suggested: SuggestedInput,
}

/// A prompt detected on a live task's screen: the class plus the screen tail
/// at detection time (what the `[waiting for input]` block echoes back).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitingPrompt {
    pub class: PromptClass,
    pub tail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptKind {
    /// Yes/no style confirmation (`[Y/n]`, `(yes/no)`, `(是/否)`, ...).
    Confirm,
    /// Numeric choice menu ("Enter choice [1-3]", "press 1/2/3", `输入编号`).
    Menu,
    /// Pager bottom (`(END)`, bare `:`, `--More--`).
    Pager,
    /// Password / passphrase / 2FA — never auto-answered at any policy tier.
    Password,
    /// SSH host-key fingerprint confirmation — stop tier.
    SshHostKey,
    /// "Press Enter to continue" and TUI dialog buttons (`<Ok>`, `[OK]`).
    PressEnter,
    /// Reserved for the (product-side) judge layer; no pattern maps here.
    Unknown,
}

impl PromptKind {
    /// Wire name for the `[waiting for input]` block and tool results.
    pub fn as_str(self) -> &'static str {
        match self {
            PromptKind::Confirm => "confirm",
            PromptKind::Menu => "menu",
            PromptKind::Pager => "pager",
            PromptKind::Password => "password",
            PromptKind::SshHostKey => "ssh_host_key",
            PromptKind::PressEnter => "press_enter",
            PromptKind::Unknown => "unknown",
        }
    }
}

/// The deterministic, safe canned answer for a class — the only input the
/// `auto` policy tier may ever send. `None` marks a class that must not be
/// auto-answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuggestedInput {
    Text(&'static str),
    Key(&'static str),
    None,
}

/// Strip ANSI escape sequences (CSI, OSC, ST forms) so patterns see readable
/// text only.
pub fn strip_ansi(text: &str) -> String {
    static ANSI_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b\\[^\x07]*\x07")
            .expect("static pattern compiles")
    });
    ANSI_RE.replace_all(text, "").to_string()
}

/// Match an **ANSI-stripped** output tail against the prompt vocabulary.
/// First match wins; the row order in [`patterns`] is the priority ladder
/// (Password above all: a tail carrying both `[Y/n]` and `Password:` reports
/// `Password`).
pub fn match_prompt(ansi_stripped_tail: &str) -> Option<PromptClass> {
    for row in ROWS {
        if row.re.is_match(ansi_stripped_tail) {
            return Some(PromptClass {
                kind: row.kind,
                default_side: row.default_side,
                suggested: row.suggested,
            });
        }
    }
    None
}

/// Last `max_chars` characters of `stdout` + `stderr` (stderr appended after
/// stdout; the interleaving order is not preserved). Char-boundary safe.
pub fn output_tail(stdout: &str, stderr: &str, max_chars: usize) -> String {
    let mut joined = String::with_capacity(stdout.len() + stderr.len());
    joined.push_str(stdout);
    if !stderr.is_empty() {
        if !joined.is_empty() {
            joined.push('\n');
        }
        joined.push_str(stderr);
    }
    joined
        .chars()
        .rev()
        .take(max_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

/// Render the `[waiting for input]` block a blocked command returns instead of
/// hanging to timeout. Wire vocabulary is English and stable — tool results
/// and tests pin its shape.
pub fn format_waiting_block(class: &PromptClass, tail: &str) -> String {
    format_waiting_block_with(
        class,
        tail,
        "the tail matches an interactive prompt and the process was killed; re-run with the answer piped in (e.g. `yes | <command>`) or a non-interactive flag (e.g. `-y`)",
    )
}

/// The `[waiting for input]` block for a live PTY task: the process is still
/// running and blocked on the prompt — the answer goes back through
/// `task_input`, not a re-run.
pub fn format_waiting_block_for_task(class: &PromptClass, tail: &str) -> String {
    format_waiting_block_with(
        class,
        tail,
        "the tail matches an interactive prompt and the task is blocked on it; send the answer with task_input (text or key)",
    )
}

/// The block with an explicit note line (the two hosts differ in what to do
/// next — kill-and-repipe vs feed-the-live-task).
pub fn format_waiting_block_with(class: &PromptClass, tail: &str, note: &str) -> String {
    let default = match class.default_side {
        // Lowercase on the wire: `default: y|n|none`.
        Some(c) => c.to_ascii_lowercase().to_string(),
        None => "none".to_string(),
    };
    let suggested = match class.suggested {
        SuggestedInput::Text(t) => t.replace('\n', "\\n"),
        SuggestedInput::Key(k) => format!("<{k}>"),
        SuggestedInput::None => "none".to_string(),
    };
    format!(
        "[waiting for input]\nclass: {}\ndefault: {}\nsuggested: {}\nnote: {}\ntail:\n{}",
        class.kind.as_str(),
        default,
        suggested,
        note,
        tail
    )
}

/// The `needs user` block a hard-boundary class returns instead of sending:
/// the secret is typed by the human at the terminal, at every policy tier.
/// `kind_name` is the class wire name (`free` when no prompt is detected).
pub fn format_needs_user_block(kind_name: &str, tail: &str) -> String {
    format!(
        "[needs user]\nclass: {}\nnote: this prompt needs a secret or a trust decision and is never auto-answered; ask the user to type it at the terminal\ntail:\n{}",
        kind_name, tail
    )
}

/// The detection notice text (TUI-visible, English wire vocabulary).
pub fn notice_text(task_id: &str, class: &PromptClass, tail: &str) -> String {
    format!(
        "task {task_id} is waiting for input ({}): {}",
        class.kind.as_str(),
        tail.lines().last().unwrap_or(tail)
    )
}

#[cfg(test)]
#[path = "interaction_tests.rs"]
mod tests;
