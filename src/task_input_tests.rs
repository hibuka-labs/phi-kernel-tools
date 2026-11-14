//! `task_input` policy tests — effects level: what actually left the tool
//! (bytes to the terminal, consent asks), not what the result text claims.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use agent_base::{
    ApprovalDecision, ApprovalRequest, AskAnswer, AskOption, Content, InteractionChannel,
    InteractionError, Tool, ToolContext,
};
use async_trait::async_trait;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::TaskInputTool;
use crate::background_shell::BackgroundTaskRegistry;
use crate::interaction::policy::{PolicyTier, PromptPolicy, SendInput};
use crate::interaction::{PromptClass, PromptKind, SuggestedInput, WaitingPrompt};
use crate::term_session::TermSession;

/// What actually reached the terminal.
#[derive(Default)]
struct SentLog {
    texts: Vec<String>,
    keys: Vec<String>,
}

struct FakeTermSession {
    sent: Arc<Mutex<SentLog>>,
}

#[async_trait]
impl TermSession for FakeTermSession {
    async fn read_screen(&self, _max_chars: usize) -> String {
        String::new()
    }
    async fn send_text(&self, text: &str) -> Result<(), String> {
        self.sent.lock().unwrap().texts.push(text.to_string());
        Ok(())
    }
    async fn send_key(&self, key: &str) -> Result<(), String> {
        self.sent.lock().unwrap().keys.push(key.to_string());
        Ok(())
    }
    async fn resize(&self, _cols: u16, _rows: u16) -> Result<(), String> {
        Ok(())
    }
    async fn close(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Records consent asks and answers with one canned decision.
struct FakeInteract {
    asked: Arc<Mutex<Vec<ApprovalRequest>>>,
    decision: ApprovalDecision,
}

#[async_trait]
impl InteractionChannel for FakeInteract {
    async fn request_consent(
        &self,
        request: ApprovalRequest,
        _cancel: &CancellationToken,
    ) -> Result<ApprovalDecision, InteractionError> {
        self.asked.lock().unwrap().push(request);
        Ok(self.decision.clone())
    }

    async fn ask_user(
        &self,
        _question: &str,
        _options: &[AskOption],
        _cancel: &CancellationToken,
    ) -> Result<AskAnswer, InteractionError> {
        Ok(AskAnswer::Skipped)
    }
}

struct Rig {
    tool: TaskInputTool,
    registry: Arc<BackgroundTaskRegistry>,
    task_id: String,
    sent: Arc<Mutex<SentLog>>,
    asked: Arc<Mutex<Vec<ApprovalRequest>>>,
}

impl Rig {
    /// Simulate detection re-firing (the program re-prompted after the last
    /// send): the waiting marker is what keys the action cache.
    fn reprompt(&self, class: PromptClass) {
        self.registry.set_waiting_input(
            &self.task_id,
            WaitingPrompt {
                class,
                tail: "Continue? [Y/n] ".to_string(),
            },
        );
    }

    async fn call(&self, args: serde_json::Value) -> (Vec<Content>, ToolContext) {
        let ctx = ToolContext::for_test();
        let out = self.tool.call(&args, &ctx).await.expect("call");
        (out, ctx)
    }

    async fn send_text(&self, text: &str) {
        let (out, ctx) = self
            .call(json!({"task_id": self.task_id, "text": text}))
            .await;
        assert!(
            !ctx.waiting_for_human.load(Ordering::Relaxed),
            "a successful send must not leave the run waiting on a human; got {out:?}"
        );
    }

    fn sent_texts(&self) -> Vec<String> {
        self.sent.lock().unwrap().texts.clone()
    }

    fn sent_keys(&self) -> Vec<String> {
        self.sent.lock().unwrap().keys.clone()
    }

    fn ask_count(&self) -> usize {
        self.asked.lock().unwrap().len()
    }
}

fn rig(class: Option<PromptClass>, policy: PromptPolicy, decision: ApprovalDecision) -> Rig {
    let sent = Arc::new(Mutex::new(SentLog::default()));
    let session: Arc<dyn TermSession> = Arc::new(FakeTermSession {
        sent: Arc::clone(&sent),
    });
    let registry = BackgroundTaskRegistry::new(8);
    let task_id = registry
        .register_pty("fixture", None, CancellationToken::new(), None, 0, session)
        .expect("register");
    if let Some(class) = class {
        registry.set_waiting_input(
            &task_id,
            WaitingPrompt {
                class,
                tail: "Continue? [Y/n] ".to_string(),
            },
        );
    }
    let asked = Arc::new(Mutex::new(Vec::new()));
    let interact = Arc::new(FakeInteract {
        asked: Arc::clone(&asked),
        decision,
    });
    let tool = TaskInputTool::new(Arc::clone(&registry))
        .with_policy(policy)
        .with_interact(interact);
    Rig {
        tool,
        registry,
        task_id,
        sent,
        asked,
    }
}

fn confirm() -> PromptClass {
    PromptClass {
        kind: PromptKind::Confirm,
        default_side: Some('Y'),
        suggested: SuggestedInput::Text("y\n"),
    }
}

fn pager() -> PromptClass {
    PromptClass {
        kind: PromptKind::Pager,
        default_side: None,
        suggested: SuggestedInput::Key("q"),
    }
}

fn text_of(out: &[Content]) -> String {
    match &out[0] {
        Content::Text { text } => text.clone(),
        other => panic!("expected text content, got {other:?}"),
    }
}

/// Auto tier: deterministic sends go straight to the terminal — zero consent
/// asks, bytes on the wire.
#[tokio::test]
async fn auto_tier_sends_directly_without_consent() {
    let rig = rig(
        Some(pager()),
        PromptPolicy::spec_default(),
        ApprovalDecision::Deny,
    );

    rig.send_text("q\n").await;
    assert_eq!(rig.sent_texts(), vec!["q\n".to_string()]);
    assert_eq!(rig.ask_count(), 0, "auto tier must never ask a human");

    // A key press on the same tier, again without a consent ask.
    rig.reprompt(pager());
    rig.call(json!({"task_id": rig.task_id, "key": "enter"}))
        .await;
    assert_eq!(rig.sent_keys(), vec!["enter".to_string()]);
    assert_eq!(rig.ask_count(), 0);
}

/// Cache tier: `AllowAlways` consents once per action key, then goes direct —
/// and a *different* answer is a different key, so it asks again.
#[tokio::test]
async fn cache_tier_allow_always_caches_per_action_key() {
    let rig = rig(
        Some(confirm()),
        PromptPolicy::spec_default(),
        ApprovalDecision::AllowAlways,
    );

    rig.send_text("y\n").await;
    assert_eq!(rig.ask_count(), 1);
    assert_eq!(rig.sent_texts(), vec!["y\n".to_string()]);
    {
        let asked = rig.asked.lock().unwrap();
        let expected = PromptPolicy::spec_default().send_action_key(
            &rig.task_id,
            Some(confirm()),
            &SendInput::Text("y\n".to_string()),
        );
        assert_eq!(asked[0].action_key, Some(expected));
    }

    // The same answer to the same class: cached, no second ask.
    rig.reprompt(confirm());
    rig.send_text("y\n").await;
    assert_eq!(rig.ask_count(), 1, "AllowAlways must cache the action key");
    assert_eq!(rig.sent_texts().len(), 2);

    // A different answer is a different action key — consent again.
    rig.reprompt(confirm());
    rig.send_text("n\n").await;
    assert_eq!(rig.ask_count(), 2);
    assert_eq!(rig.sent_texts().len(), 3);
}

/// Cache tier with `AllowOnce`: nothing is cached, every send asks.
#[tokio::test]
async fn cache_tier_allow_once_never_caches() {
    let rig = rig(
        Some(confirm()),
        PromptPolicy::spec_default(),
        ApprovalDecision::AllowOnce,
    );

    rig.send_text("y\n").await;
    rig.reprompt(confirm());
    rig.send_text("y\n").await;
    assert_eq!(rig.ask_count(), 2);
    assert_eq!(rig.sent_texts().len(), 2);
}

/// Hard boundary: password / ssh-host-key / unknown return `[needs user]` and
/// send **zero bytes even at an all-Auto table** — no consent ask either, and
/// the run is marked waiting on the human.
#[tokio::test]
async fn stop_tier_sends_nothing_even_at_all_auto_table() {
    let all_auto = PromptPolicy {
        pager: PolicyTier::Auto,
        confirm: PolicyTier::Auto,
        menu: PolicyTier::Auto,
        press_enter: PolicyTier::Auto,
        free_input: PolicyTier::Auto,
    };
    let secret_classes = [
        PromptClass {
            kind: PromptKind::Password,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        PromptClass {
            kind: PromptKind::SshHostKey,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        PromptClass {
            kind: PromptKind::Unknown,
            default_side: None,
            suggested: SuggestedInput::None,
        },
    ];

    let rig = rig(
        Some(secret_classes[0]),
        all_auto,
        ApprovalDecision::AllowAlways,
    );
    for class in &secret_classes {
        rig.reprompt(*class);
        let (out, ctx) = rig
            .call(json!({"task_id": rig.task_id, "text": "hunter2\n"}))
            .await;
        let text = text_of(&out);
        assert!(
            text.starts_with("[needs user]"),
            "class {:?} must return needs-user, got: {text}",
            class.kind
        );
        assert!(
            ctx.waiting_for_human.load(Ordering::Relaxed),
            "class {:?} must leave the run waiting on a human",
            class.kind
        );
        assert_eq!(
            rig.sent_texts().len(),
            0,
            "zero bytes may leave the tool, iteration {}",
            class.kind.as_str()
        );
        assert_eq!(
            rig.ask_count(),
            0,
            "a secret is never consented, it is refused"
        );
    }
}

/// Wire union: exactly one of `text` / `key`, key names from the vocabulary.
#[tokio::test]
async fn wire_rejects_bad_input_shapes() {
    let rig = rig(
        Some(pager()),
        PromptPolicy::spec_default(),
        ApprovalDecision::Deny,
    );

    for (args, want) in [
        (
            json!({"task_id": rig.task_id, "text": "a", "key": "enter"}),
            "exactly one of",
        ),
        (json!({"task_id": rig.task_id}), "provide one of"),
        (json!({"task_id": rig.task_id, "key": "f1"}), "unknown key"),
    ] {
        let (out, _) = rig.call(args).await;
        let text = text_of(&out);
        assert!(text.contains(want), "want {want:?} in: {text}");
    }
    assert!(rig.sent_texts().is_empty() && rig.ask_count() == 0);
}
