use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, ToolControlFlow, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
pub struct WaitAgentArgs {
    #[serde(default)]
    pub agent_path: Option<String>,
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

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_path": {
                    "type": "string",
                    "description": "Optional: specific agent to wait for. Omit for any."
                },
                "timeout_ms": {
                    "type": "number",
                    "description": "Max wait time in ms (default: 120000 = 2 min)"
                }
            },
            "required": []
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
        let result = self
            .runtime
            .wait_for_result(args.agent_path.as_deref(), args.timeout_ms)
            .await;

        Ok(WaitAgentOutput {
            status: result.status,
            result: result.result,
            agent_path: result.agent_path,
            has_more: result.has_more,
        })
    }
}
