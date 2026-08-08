use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, ToolControlFlow, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
pub struct FollowupTaskArgs {
    pub agent_path: String,
    pub task: String,
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

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_path": {
                    "type": "string",
                    "description": "Target agent path (e.g., 'root/searcher')"
                },
                "task": {
                    "type": "string",
                    "description": "Task description for the sub-agent"
                },
                "interrupt": {
                    "type": "boolean",
                    "description": "Whether to interrupt current task (default: true)"
                }
            },
            "required": ["agent_path", "task"]
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
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
