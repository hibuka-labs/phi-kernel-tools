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
        "Send a message to a sub-agent.\n\
         By default the message is queued as context and does not trigger\n\
         execution; set trigger=true to hand it over as a task the child\n\
         will run (tasks run serially inside a child).\n\
         Returns delivered=false if the child is gone or the queue is full."
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
