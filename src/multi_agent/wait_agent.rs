use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct WaitAgentArgs {
    /// Optional: specific agent to wait for. Omit for any.
    #[serde(default)]
    pub agent_path: Option<String>,
    /// Max wait time in ms (default: 120000 = 2 min)
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    120_000
}

#[derive(Debug, Serialize)]
pub struct WaitAgentOutput {
    pub status: String,
    pub result: Option<String>,
    pub agent_path: Option<String>,
    pub has_more: bool,
    /// Tools the child attempted but was denied permission to call.
    pub denied_tools: Vec<String>,
}

pub struct WaitAgentTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl WaitAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for WaitAgentTool {
    type Args = WaitAgentArgs;
    type Output = WaitAgentOutput;

    fn name(&self) -> &'static str {
        "wait_agent"
    }

    fn description(&self) -> &'static str {
        "Wait for a sub-agent to complete and return its result.\n\
         If agent_path is omitted, waits for ANY sub-agent.\n\
         Returns timeout if no agent completes within the timeout.\n\
         Check has_more for additional pending results."
    }

    async fn call_typed(&self, args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        let result = self
            .runtime
            .wait_for_result(args.agent_path.as_deref(), args.timeout_ms)
            .await;

        Ok(WaitAgentOutput {
            status: result.status,
            result: result.result,
            agent_path: result.agent_path,
            has_more: result.has_more,
            denied_tools: result.denied_tools,
        })
    }
}
