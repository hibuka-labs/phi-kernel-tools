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
        "Send a message to a sub-agent without triggering execution.\n\
         The message is queued and delivered with the next followup_task."
    }

    async fn call_typed(&self, args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        let delivered = self
            .runtime
            .send_message(&args.agent_path, args.message)
            .unwrap_or(false);
        Ok(SendMessageOutput { delivered })
    }
}
