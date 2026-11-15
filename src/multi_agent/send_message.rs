use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SendMessageArgs {
    /// Target agent path (e.g., 'root/searcher')
    pub agent_path: String,
    /// Message content to deliver
    pub message: String,
    /// Also trigger execution: the message is queued as a task and the child
    /// starts working on it. Default false = context only, delivered with
    /// the child's next task. (This is the qualified replacement for the
    /// deprecated `followup_task` tool — design doc §8.3.)
    #[serde(default)]
    pub trigger: bool,
}

#[derive(Debug, Serialize)]
pub struct SendMessageOutput {
    pub delivered: bool,
}

/// The face of `send_message`.
///
/// Carries when to send as much as how: a follow-up that only asks for
/// progress cannot make a child finish sooner, and the cost lands on whoever
/// is waiting for the batch.
const DESCRIPTION: &str = "\
Send a message to a sub-agent.\n\
By default the message is queued as context and does not trigger\n\
execution; set trigger=true to hand it over as a task the child\n\
will run (tasks run serially inside a child).\n\
Returns delivered=false if the child is gone or the queue is full.\n\
Send when you have context that changes the task or a course correction\n\
worth acting on. Do NOT send a progress nudge — the child is still working,\n\
and asking it to wrap up cannot make it finish sooner; its report arrives when\n\
it arrives, whether you watch or not. A message is not a faster result.\n\
Prefer this over closing and re-spawning: a message keeps the child's\n\
context, a new spawn starts it clean.";

pub struct SendMessageTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl SendMessageTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for SendMessageTool {
    type Args = SendMessageArgs;
    type Output = SendMessageOutput;

    fn name(&self) -> &'static str {
        "send_message"
    }

    fn description(&self) -> &'static str {
        DESCRIPTION
    }

    async fn call_typed(&self, args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        // trigger=true is `send_task`; the old `followup_task` trigger path.
        // Tasks are serial inside a child (defect K2): no interrupt semantics.
        let delivered = if args.trigger {
            self.runtime
                .send_task(&args.agent_path, args.message, false)
        } else {
            self.runtime.send_message(&args.agent_path, args.message)
        }
        .unwrap_or(false);
        Ok(SendMessageOutput { delivered })
    }
}

#[cfg(test)]
mod tests {

    /// Value: protects=the child's time — a nudge cannot make it finish
    /// sooner, and the cost lands on whoever is waiting.
    /// why_new=the face said what `trigger` does and never when sending is
    /// the wrong move.
    /// seam=none; no_claim=that the model respects it.
    #[test]
    fn description_says_when_not_to_send() {
        let d = super::DESCRIPTION;
        assert!(
            d.contains("nudge") || d.contains("still working"),
            "a follow-up tool must say a progress nudge is not one: {d}"
        );
        assert!(
            d.contains("context") || d.contains("course"),
            "and what it is for — added context or a change of direction: {d}"
        );
    }
}
