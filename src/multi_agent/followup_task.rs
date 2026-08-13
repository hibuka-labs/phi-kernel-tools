use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FollowupTaskArgs {
    /// Target agent path (e.g., 'root/searcher')
    pub agent_path: String,
    /// Task description for the sub-agent
    pub task: String,
    /// Whether to interrupt current task (default: true)
    #[serde(default = "default_interrupt")]
    pub interrupt: bool,
}

fn default_interrupt() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct FollowupTaskOutput {
    pub accepted: bool,
    pub agent_path: String,
}

pub struct FollowupTaskTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl FollowupTaskTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for FollowupTaskTool {
    type Args = FollowupTaskArgs;
    type Output = FollowupTaskOutput;

    fn name(&self) -> &'static str {
        "followup_task"
    }

    fn description(&self) -> &'static str {
        "Send a task to a sub-agent and trigger execution.\n\
         Returns immediately. Use wait_agent to collect results.\n\
         Set interrupt=false to queue after current task completes."
    }

    async fn call_typed(&self, args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        match self
            .runtime
            .send_task(&args.agent_path, args.task, args.interrupt)
        {
            Ok(accepted) => Ok(FollowupTaskOutput {
                accepted,
                agent_path: args.agent_path,
            }),
            Err(e) => Ok(FollowupTaskOutput {
                accepted: false,
                agent_path: format!("error: {}", e),
            }),
        }
    }
}
