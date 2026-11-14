//! `task_input` — send input to a running PTY task.
//!
//! The model sees a blocked interactive prompt through `task_output`'s
//! `[waiting for input]` block, decides what to answer (the "AI reads the
//! screen" main path), and sends it here. The answer policy sits on top:
//!
//! - [`PolicyTier::Auto`]: deterministic safe answer — sent directly;
//! - [`PolicyTier::Cache`]: first send per action key goes through the
//!   product's consent channel, `AllowAlways` caches the key and later sends
//!   go direct;
//! - [`PolicyTier::Stop`] (the hard boundary: password / 2FA / SSH host key /
//!   unknown): **zero bytes leave this tool** — it returns `[needs user]`.
//!
//! The consent channel is `agent_base`'s [`InteractionChannel`] seam,
//! injected at build time; the product decides how a human nods. Without a
//! mounted channel a cache-tier send fails closed (never sends unasked).

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use agent_base::{
    AgentResult, ApprovalDecision, ApprovalRequest, Content, InteractionChannel, InteractionError,
    Tool, ToolContext,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::background_shell::BackgroundTaskRegistry;
use crate::interaction::{
    format_needs_user_block,
    policy::{KEY_NAMES, PolicyTier, PromptPolicy, SendInput},
};
use crate::task_registry::TaskStatus;

/// Suspension discipline for a consent ask (mirrors `ExecCtx::guard` at the
/// tool's own scale): an unanswered ask must not deadlock the run — the run's
/// cancel token first, then the interaction watchdog.
const INTERACT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// `task_input` — feed a running PTY task's terminal.
#[derive(Clone)]
pub struct TaskInputTool {
    registry: Arc<BackgroundTaskRegistry>,
    policy: PromptPolicy,
    interact: Option<Arc<dyn InteractionChannel>>,
    /// `AllowAlways` cache: action keys a human has stood behind this session.
    allowed: Arc<Mutex<HashSet<String>>>,
}

impl TaskInputTool {
    pub fn new(registry: Arc<BackgroundTaskRegistry>) -> Self {
        Self {
            registry,
            policy: PromptPolicy::spec_default(),
            interact: None,
            allowed: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Product-filled answer-policy table (mechanism default: spec mapping).
    pub fn with_policy(mut self, policy: PromptPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Product's consent channel for cache-tier sends.
    pub fn with_interact(mut self, interact: Arc<dyn InteractionChannel>) -> Self {
        self.interact = Some(interact);
        self
    }

    /// Parse the wire union (`text` XOR `key`) into a [`SendInput`].
    fn parse_input(args: &Value) -> Result<SendInput, String> {
        let text = args.get("text").and_then(Value::as_str);
        let key = args.get("key").and_then(Value::as_str);
        match (text, key) {
            (Some(t), None) => Ok(SendInput::Text(t.to_string())),
            (None, Some(k)) => {
                if KEY_NAMES.contains(&k) {
                    Ok(SendInput::Key(k.to_string()))
                } else {
                    Err(format!(
                        "[Error]: unknown key {k:?}; valid keys: {}",
                        KEY_NAMES.join(", ")
                    ))
                }
            }
            (Some(_), Some(_)) => {
                Err("[Error]: provide exactly one of `text` or `key`.".to_string())
            }
            (None, None) => Err("[Error]: provide one of `text` or `key`.".to_string()),
        }
    }

    /// The cache-tier ask: direct if the key was already allowed, otherwise
    /// suspend on the consent channel with the suspension discipline.
    async fn authorize_cached(
        &self,
        request: ApprovalRequest,
        cancel: &CancellationToken,
    ) -> Result<Consent, String> {
        let key = request.action_key.clone().unwrap_or_default();
        if self.allowed.lock().unwrap().contains(&key) {
            return Ok(Consent::Cached);
        }
        let Some(interact) = &self.interact else {
            return Err(
                "[Error]: cache-tier input needs a consent channel; none is mounted. \
                 The product layer must mount one at build time."
                    .to_string(),
            );
        };
        let decision = tokio::select! {
            res = interact.request_consent(request, cancel) => res.map_err(|e| match e {
                InteractionError::Cancelled => "[Error]: consent cancelled.".to_string(),
                InteractionError::Timeout(_) => "[Error]: consent timed out.".to_string(),
                InteractionError::Channel(msg) => format!("[Error]: consent failed: {msg}"),
            })?,
            _ = cancel.cancelled() => return Err("[Error]: consent cancelled.".to_string()),
            _ = tokio::time::sleep(INTERACT_TIMEOUT) => {
                return Err("[Error]: consent timed out.".to_string());
            }
        };
        match decision {
            ApprovalDecision::AllowAlways => {
                self.allowed.lock().unwrap().insert(key);
                Ok(Consent::Approved)
            }
            ApprovalDecision::AllowOnce => Ok(Consent::Approved),
            ApprovalDecision::Deny => Ok(Consent::Denied),
        }
    }
}

/// How a send was authorized — the result text says which, so the transcript
/// never claims a human approved something they did not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Consent {
    /// A human approved this send just now.
    Approved,
    /// The same answer was approved earlier ("always") — the cache hit.
    Cached,
    /// A human refused.
    Denied,
}

#[async_trait]
impl Tool for TaskInputTool {
    fn name(&self) -> &'static str {
        "task_input"
    }

    fn description(&self) -> &'static str {
        "Send input to a running PTY task (from execute_command with pty: true).\n\
         Use it to answer an interactive prompt reported by task_output's `[waiting for input]` block:\n\
         look at the screen tail, decide the answer by task intent (y/n, a menu number, a key), and send it.\n\
         Exactly one of `text` (include \\n explicitly when the program reads a line) or `key`.\n\
         Passwords, 2FA codes and SSH host-key confirmations are NEVER accepted here — they return\n\
         `[needs user]` and must be typed by the human at the terminal."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "PTY task ID from execute_command (bg_* format)."
                },
                "text": {
                    "type": "string",
                    "description": "Text to send; include \\n explicitly when needed."
                },
                "key": {
                    "type": "string",
                    "enum": KEY_NAMES,
                    "description": "Named key press to send instead of text."
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
        // A cache-tier send may suspend on human consent; the tool fuse must
        // sit above the interaction watchdog ([`INTERACT_TIMEOUT`]).
        Some(INTERACT_TIMEOUT.as_millis() as u64 + 30_000)
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
        let input = match Self::parse_input(args) {
            Ok(input) => input,
            Err(msg) => return Ok(vec![Content::text(msg)]),
        };

        let Some(snap) = self.registry.snapshot(&task_id) else {
            return Ok(vec![Content::text(format!(
                "[Error]: Task {} not found (finished tasks are cleaned up after their output is fetched).",
                task_id
            ))]);
        };
        if !snap.pty {
            return Ok(vec![Content::text(format!(
                "[Error]: Task {} is not a PTY task; only PTY tasks (execute_command with pty: true) can receive input.",
                task_id
            ))]);
        }
        if snap.status.is_terminal() {
            return Ok(vec![Content::text(format!(
                "[Error]: Task {} already finished with status: {:?}.",
                task_id, snap.status
            ))]);
        }

        // The prompt class the task is blocked on (None = free typing). Only
        // the mechanism's hard boundary can map to Stop here without a policy
        // override — see [`PromptPolicy::tier_for`].
        let waiting = snap.waiting_input.clone();
        let class = waiting.as_ref().map(|w| w.class);
        let tail = waiting.as_ref().map(|w| w.tail.clone()).unwrap_or_default();

        if self.policy.tier_for(class) == PolicyTier::Stop {
            // Hard boundary: zero bytes leave this tool. The run waits on a
            // human at the terminal.
            ctx.mark_waiting_for_human();
            let kind = class.map(|c| c.kind.as_str()).unwrap_or("free");
            return Ok(vec![Content::text(format_needs_user_block(kind, &tail))]);
        }

        // How the send was authorized — the result says so, so the transcript
        // never claims a human approved something they did not. `None` = the
        // auto tier: nobody was asked.
        let consent = if self.policy.tier_for(class) == PolicyTier::Cache {
            let request = self.policy.approval_request(&task_id, class, &input);
            // Pause the call's execution clock while a human is asked.
            let _pause = ctx.pause_execution_clock();
            match self.authorize_cached(request, &ctx.cancel_token).await {
                Ok(Consent::Approved) => Some(Consent::Approved),
                Ok(Consent::Cached) => Some(Consent::Cached),
                Ok(Consent::Denied) => {
                    return Ok(vec![Content::text(format!(
                        "[denied] input for task {} was not approved; nothing was sent.",
                        task_id
                    ))]);
                }
                Err(msg) => return Ok(vec![Content::text(msg)]),
            }
        } else {
            None
        };

        // Auto tier (and cached/consented sends): deliver the bytes.
        let Some(session) = self.registry.pty_session(&task_id) else {
            return Ok(vec![Content::text(format!(
                "[Error]: Task {} has no live PTY session.",
                task_id
            ))]);
        };
        let result = match &input {
            SendInput::Text(t) => session.send_text(t).await,
            SendInput::Key(k) => session.send_key(k).await,
        };
        match result {
            Ok(()) => {
                // The prompt was answered (or a key went out): detection
                // re-arms from fresh output, so drop the stale marker now.
                self.registry.clear_waiting_input(&task_id);
                let approved_by = match consent {
                    Some(Consent::Approved) => " (approved by the user)",
                    Some(Consent::Cached) => " (approved earlier for this answer)",
                    Some(Consent::Denied) | None => "",
                };
                Ok(vec![Content::text(format!(
                    "sent {} to task {}{approved_by} — read the response with task_output.",
                    input.action_repr(),
                    task_id
                ))])
            }
            Err(e) => Ok(vec![Content::text(format!("[Error]: send failed: {}", e))]),
        }
    }
}

#[cfg(test)]
#[path = "task_input_tests.rs"]
mod tests;
