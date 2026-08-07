use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, ToolControlFlow, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
pub struct SendMessageArgs {
    pub agent_path: String,
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

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_path": {
                    "type": "string",
                    "description": "Target agent path (e.g., 'root/searcher')"
                },
                "message": {
                    "type": "string",
                    "description": "Message content to deliver"
                }
            },
            "required": ["agent_path", "message"]
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(
        &self,
        args: Self::Args,
        _ctx: &ToolContext,
    ) -> AgentResult<Self::Output> {
        let delivered = self
            .runtime
            .send_message(&args.agent_path, args.message)
            .unwrap_or(false);
        Ok(SendMessageOutput { delivered })
    }
}
