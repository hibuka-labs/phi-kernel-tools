//! Answer-policy mapping: which tier governs a send into a waiting prompt.
//!
//! The policy is a table the product fills ([`PromptPolicy`]); this module
//! holds the mechanism's rules on top of it:
//!
//! - the three tiers ([`PolicyTier`]): `Auto` (deterministic safe answer,
//!   direct send), `Cache` (the model picks, a human nods once per
//!   [`PromptPolicy::send_action_key`], later sends go direct), `Stop` (never send —
//!   return `needs user`);
//! - the hard boundary: Password / 2FA / SshHostKey / Unknown always map to
//!   `Stop`, whatever the table says. Secrets are typed by the human at the
//!   terminal, at every policy tier.

use agent_base::{ApprovalRequest, RiskLevel};

use super::{PromptClass, PromptKind, SuggestedInput};

/// Policy tier for one prompt class (answer-policy axis: ask a human or not).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyTier {
    /// Deterministic safe answer — send directly, never ask a human.
    Auto,
    /// The model (or a human) picks the input; first send per action key goes
    /// through consent, `AllowAlways` caches it and later sends go direct.
    Cache,
    /// Never send — the tool returns `needs user`.
    Stop,
}

/// The answer-policy table. Product-filled; the mechanism only reads it.
///
/// Defaults follow the spec's approved mapping: pager quits are deterministic
/// (`Auto`), confirm/menu/press-enter picks are the model's to make but a
/// human nods the first one (`Cache`), and free input (no prompt class on
/// screen) also sits at `Cache` — relaxing `free_input` to `Auto` is the one
/// line a product turns when it trusts the model with plain keystrokes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptPolicy {
    pub pager: PolicyTier,
    pub confirm: PolicyTier,
    pub menu: PolicyTier,
    pub press_enter: PolicyTier,
    /// Tier for a send while no prompt class is detected (free typing).
    pub free_input: PolicyTier,
}

impl Default for PromptPolicy {
    fn default() -> Self {
        Self {
            pager: PolicyTier::Auto,
            confirm: PolicyTier::Cache,
            menu: PolicyTier::Cache,
            press_enter: PolicyTier::Cache,
            free_input: PolicyTier::Cache,
        }
    }
}

impl PromptPolicy {
    /// The spec's approved table (same as [`Default`]).
    pub fn spec_default() -> Self {
        Self::default()
    }

    /// Tier for the class currently on screen (`None` = free input).
    ///
    /// The hard boundary is enforced here and nowhere else: Password, 2FA,
    /// SSH host keys and the unknown class map to [`PolicyTier::Stop`] even
    /// when the table says otherwise. `SshHostKey` is a stop even though its
    /// text carries a yes/no question — the answer is a trust decision for
    /// the human at the terminal.
    pub fn tier_for(&self, class: Option<PromptClass>) -> PolicyTier {
        match class.map(|c| c.kind) {
            Some(PromptKind::Password)
            | Some(PromptKind::SshHostKey)
            | Some(PromptKind::Unknown) => PolicyTier::Stop,
            Some(PromptKind::Pager) => self.pager,
            Some(PromptKind::Confirm) => self.confirm,
            Some(PromptKind::Menu) => self.menu,
            Some(PromptKind::PressEnter) => self.press_enter,
            None => self.free_input,
        }
    }

    /// Action key for the consent cache: one key per (task, class, input), so
    /// `AllowAlways` on one answer never pre-approves a different answer.
    pub fn send_action_key(
        &self,
        task_id: &str,
        class: Option<PromptClass>,
        input: &SendInput,
    ) -> String {
        let kind = class.map(|c| c.kind.as_str()).unwrap_or("free");
        format!("task_input:{task_id}:{kind}:{}", input.action_repr())
    }

    /// The consent request for a cache-tier send (title/message/action_key).
    /// Built here so every host shows the same picture and keys the cache the
    /// same way.
    pub fn approval_request(
        &self,
        task_id: &str,
        class: Option<PromptClass>,
        input: &SendInput,
    ) -> ApprovalRequest {
        let kind = class.map(|c| c.kind.as_str()).unwrap_or("free");
        ApprovalRequest {
            // The picture a human reads: what is waiting, what the agent
            // wants to answer, and which prompt class it is. The task id is
            // the handle the human uses to find the task.
            title: format!("Task {task_id} is waiting for input"),
            message: format!(
                "The task is blocked on a {kind} prompt. Sending {} resumes it.",
                input.human_repr()
            ),
            // The cache key: one per (task, class, answer), so "always" on
            // one answer never pre-approves a different answer.
            action_key: Some(self.send_action_key(task_id, class, input)),
            // Not "safe": answering an interactive prompt speaks for the
            // human at a decision point.
            risk_level: RiskLevel::Sensitive,
            raw: None,
            source: None,
        }
    }
}

/// What the caller wants to send to the task's terminal — the wire union of
/// `task_input`'s `text` / `key` (key names are [`KEY_NAMES`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SendInput {
    /// Literal text; a trailing newline must be included explicitly.
    Text(String),
    /// A named key press ([`KEY_NAMES`] vocabulary on the wire).
    Key(String),
}

impl SendInput {
    /// Bytes that reach the terminal.
    pub fn wire_bytes(&self) -> Vec<u8> {
        match self {
            SendInput::Text(t) => t.as_bytes().to_vec(),
            SendInput::Key(k) => key_bytes(k),
        }
    }

    /// Stable action-key / dialog fragment (never the raw secret — the class
    /// is `Stop` for those anyway, so this only ever sees navigational input).
    pub fn action_repr(&self) -> String {
        match self {
            SendInput::Text(t) => format!("text:{t:?}"),
            SendInput::Key(k) => format!("key:{k}"),
        }
    }

    /// The same input, worded for a human reading a popup (the cache key
    /// spelling `text:"y\n"` is a key, not a sentence).
    pub fn human_repr(&self) -> String {
        match self {
            SendInput::Text(t) => format!("text {t:?}"),
            SendInput::Key(k) => format!("key {k}"),
        }
    }
}

/// Wire names of the `key` enum in `task_input`'s schema. Kept here so the
/// schema, [`SendInput::wire_bytes`] and the tests share one vocabulary.
pub const KEY_NAMES: [&str; 9] = [
    "enter", "tab", "up", "down", "left", "right", "ctrl_c", "ctrl_d", "escape",
];

/// Key name → terminal bytes. Enter is CR (what a PTY line discipline wants),
/// arrows are CSI sequences.
pub fn key_bytes(name: &str) -> Vec<u8> {
    match name {
        "enter" => b"\r".to_vec(),
        "tab" => b"\t".to_vec(),
        "up" => b"\x1b[A".to_vec(),
        "down" => b"\x1b[B".to_vec(),
        "right" => b"\x1b[C".to_vec(),
        "left" => b"\x1b[D".to_vec(),
        "ctrl_c" => b"\x03".to_vec(),
        "ctrl_d" => b"\x04".to_vec(),
        "escape" => b"\x1b".to_vec(),
        other => other.as_bytes().to_vec(),
    }
}

/// Which suggested canned answer (if any) belongs to a class — re-exported
/// convenience for hosts rendering the `[waiting for input]` block.
pub fn suggested_of(class: PromptClass) -> SuggestedInput {
    class.suggested
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
