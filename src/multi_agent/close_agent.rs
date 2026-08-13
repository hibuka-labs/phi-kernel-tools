use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CloseAgentArgs {
    /// Agent path to close (e.g., 'root/searcher')
    pub agent_path: String,
}

#[derive(Debug, Serialize)]
pub struct CloseAgentOutput {
    pub closed: bool,
    pub previous_status: String,
    pub message: String,
}

pub struct CloseAgentTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl CloseAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for CloseAgentTool {
    type Args = CloseAgentArgs;
    type Output = CloseAgentOutput;

    fn name(&self) -> &'static str {
        "close_agent"
    }

    fn description(&self) -> &'static str {
        "Close a sub-agent and release its resources.\n\
         Immediately stops the agent (aborts current task) and removes it.\n\
         Pending wait_agent calls for this agent return status='closed'."
    }

    async fn call_typed(&self, args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        match self.runtime.close_agent(&args.agent_path) {
            Ok(result) => Ok(CloseAgentOutput {
                closed: result.closed,
                previous_status: result.previous_status,
                message: result.message,
            }),
            Err(e) => Ok(CloseAgentOutput {
                closed: false,
                previous_status: "unknown".to_string(),
                message: e,
            }),
        }
    }
}
