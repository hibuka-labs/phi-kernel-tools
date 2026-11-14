//! Interactive-prompt vocabulary: the pattern rows [`super::match_prompt`]
//! scans, in priority order.
//!
//! The row layout *is* the priority ladder (first match wins):
//! Password > SshHostKey > Pager > Confirm > Menu > PressEnter.
//!
//! Every row reports shape only — `kind`, `default_side`, `suggested`. It never
//! decides what to send. `SuggestedInput` is the deterministic, safe canned
//! answer for the class (the `auto` policy tier may send exactly that);
//! `None` means the class must never be auto-answered — a decider (the model
//! reading the screen, or a human) picks the input.

use std::sync::LazyLock;

use regex::Regex;

use super::{PromptKind, SuggestedInput};

pub(crate) struct PatternRow {
    pub kind: PromptKind,
    pub default_side: Option<char>,
    pub suggested: SuggestedInput,
    pub re: LazyLock<Regex>,
}

/// One named `static` per row: a `LazyLock` (interior mutability) may not live
/// as a temporary inside another static's initializer (E0492), so every row
/// gets an addressable home and [`ROWS`] only holds references.
macro_rules! row_def {
    ($name:ident, $kind:expr, $default:expr, $suggested:expr, $pat:expr) => {
        static $name: PatternRow = PatternRow {
            kind: $kind,
            default_side: $default,
            suggested: $suggested,
            re: LazyLock::new(|| Regex::new($pat).expect("static pattern compiles")),
        };
    };
}

// ── Password (top: a password prompt beats every other class) ──
row_def!(
    ROW_PASSWORD,
    PromptKind::Password,
    None,
    SuggestedInput::None,
    r"(?im)^\s*(enter\s+)?(password|passphrase)\s*[:：]"
);
row_def!(
    ROW_SUDO_PASSWORD,
    PromptKind::Password,
    None,
    SuggestedInput::None,
    r"(?im)^\[sudo\] password for "
);
row_def!(
    ROW_KEY_PASSPHRASE,
    PromptKind::Password,
    None,
    SuggestedInput::None,
    r"(?im)^enter passphrase for key "
);
// 2FA / OTP prompts are secrets too: stop, never send.
row_def!(
    ROW_MFA_CODE,
    PromptKind::Password,
    None,
    SuggestedInput::None,
    r"(?im)^\s*(verification code|two[- ]factor|2fa|one[- ]time (password|code|passcode)|otp)\b[^\n]{0,40}[:：]"
);
// ── SshHostKey (above Confirm: its text carries a yes/no question) ──
row_def!(
    ROW_SSH_HOST_KEY,
    PromptKind::SshHostKey,
    None,
    SuggestedInput::None,
    r"(?i)are you sure you want to continue connecting"
);
// ── Pager ──
row_def!(
    ROW_PAGER_END,
    PromptKind::Pager,
    None,
    SuggestedInput::Text("q\n"),
    r"(?m)^\(END\)\s*$"
);
row_def!(
    ROW_PAGER_COLON,
    PromptKind::Pager,
    None,
    SuggestedInput::Text("q\n"),
    r"(?m)^:\s*$"
);
row_def!(
    ROW_PAGER_MORE,
    PromptKind::Pager,
    None,
    SuggestedInput::Text("q\n"),
    r"(?m)^--More--(\s*\(\d+%\))?\s*$"
);
// ── Confirm ──
// `[Y/n]` / `(Y/n)`: Y is the highlighted default → the deterministic
// safe answer is the default itself.
row_def!(
    ROW_CONFIRM_DEFAULT_Y,
    PromptKind::Confirm,
    Some('Y'),
    SuggestedInput::Text("y\n"),
    r"[\[(]\s*Y\s*/\s*n\s*[\])]"
);
// `[y/N]` / `(y/N)`: the default is N — never auto-accept past it.
row_def!(
    ROW_CONFIRM_DEFAULT_N,
    PromptKind::Confirm,
    Some('N'),
    SuggestedInput::None,
    r"[\[(]\s*y\s*/\s*N\s*[\])]"
);
// Neutral bracket family `[y/n]` `[Y/N]` `(y/n)`: no highlighted default.
row_def!(
    ROW_CONFIRM_NEUTRAL,
    PromptKind::Confirm,
    None,
    SuggestedInput::None,
    r"(?i)[\[(]\s*y\s*/\s*n\s*[\])]"
);
row_def!(
    ROW_CONFIRM_YESNO,
    PromptKind::Confirm,
    None,
    SuggestedInput::None,
    r"(?i)[\[(]\s*yes\s*/\s*no\s*[\])]"
);
// CJK confirm family `(是/否)`.
row_def!(
    ROW_CONFIRM_CJK,
    PromptKind::Confirm,
    None,
    SuggestedInput::None,
    r"[\[(（]\s*是\s*[/／|]\s*否\s*[\]）)]"
);
// ── Menu (numeric choices: the pick is task-intent, never canned) ──
row_def!(
    ROW_MENU_ENTER_CHOICE,
    PromptKind::Menu,
    None,
    SuggestedInput::None,
    r"(?i)enter\s+(your\s+)?(choice|option|selection|number)"
);
row_def!(
    ROW_MENU_PRESS_DIGITS,
    PromptKind::Menu,
    None,
    SuggestedInput::None,
    r"(?i)(press|type|input|enter)\s+\d+(\s*[/,|]\s*\d+)+"
);
row_def!(
    ROW_MENU_BARE_BRACKET,
    PromptKind::Menu,
    None,
    SuggestedInput::None,
    r"(?m)^\s*\[\d+(\s*[/,|]\s*\d+)+\]\s*$"
);
row_def!(
    ROW_MENU_CJK,
    PromptKind::Menu,
    None,
    SuggestedInput::None,
    r"输入(编号|数字|选项)"
);
// ── PressEnter (incl. TUI dialog buttons: Enter activates the focus) ──
row_def!(
    ROW_PRESS_ENTER,
    PromptKind::PressEnter,
    None,
    SuggestedInput::Key("enter"),
    r"(?i)press\s+(enter|return|any\s+key)\s+to\s+(continue|proceed|exit|quit|retry)"
);
row_def!(
    ROW_PRESS_ENTER_CJK,
    PromptKind::PressEnter,
    None,
    SuggestedInput::Key("enter"),
    r"按(回车|Enter)(键)?继续"
);
row_def!(
    ROW_TUI_BUTTON,
    PromptKind::PressEnter,
    None,
    SuggestedInput::Key("enter"),
    r"(?i)<\s*(ok|yes)\s*>"
);
row_def!(
    ROW_OK_BUTTON,
    PromptKind::PressEnter,
    None,
    SuggestedInput::Key("enter"),
    r"(?i)\[\s*ok\s*\]"
);

pub(crate) static ROWS: &[&PatternRow] = &[
    &ROW_PASSWORD,
    &ROW_SUDO_PASSWORD,
    &ROW_KEY_PASSPHRASE,
    &ROW_MFA_CODE,
    &ROW_SSH_HOST_KEY,
    &ROW_PAGER_END,
    &ROW_PAGER_COLON,
    &ROW_PAGER_MORE,
    &ROW_CONFIRM_DEFAULT_Y,
    &ROW_CONFIRM_DEFAULT_N,
    &ROW_CONFIRM_NEUTRAL,
    &ROW_CONFIRM_YESNO,
    &ROW_CONFIRM_CJK,
    &ROW_MENU_ENTER_CHOICE,
    &ROW_MENU_PRESS_DIGITS,
    &ROW_MENU_BARE_BRACKET,
    &ROW_MENU_CJK,
    &ROW_PRESS_ENTER,
    &ROW_PRESS_ENTER_CJK,
    &ROW_TUI_BUTTON,
    &ROW_OK_BUTTON,
];
