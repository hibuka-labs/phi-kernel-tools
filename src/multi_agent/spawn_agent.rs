use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, ToolControlFlow, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
pub struct SpawnAgentArgs {
    pub task_name: String,
    pub message: String,
    #[serde(default)]
    pub agent_type: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub fork_history: Option<String>,
    /// Nesting depth for this agent. 1 = direct child of root, 2 = grandchild, etc.
    /// Defaults to 1 (direct child).
    #[serde(default = "default_depth")]
    pub depth: i32,
}

fn default_depth() -> i32 {
    1
}

#[derive(Debug, Serialize)]
pub struct SpawnAgentOutput {
    pub agent_path: String,
    pub message: String,
}

pub struct SpawnAgentTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl SpawnAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for SpawnAgentTool {
    type Args = SpawnAgentArgs;
    type Output = SpawnAgentOutput;

    fn name(&self) -> &'static str {
        "spawn_agent"
    }

    fn description(&self) -> &'static str {
        "Spawn a new sub-agent to execute a task independently.\n\
         The sub-agent runs concurrently and reports results via its mailbox.\n\
         Use agent_type for preset roles, or system_prompt for custom instructions."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_name": {
                    "type": "string",
                    "description": "Unique name for this sub-agent (used in agent path)"
                },
                "message": {
                    "type": "string",
                    "description": "Initial task description for the sub-agent"
                },
                "agent_type": {
                    "type": "string",
                    "description": "Optional role type that maps to a preset configuration"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional custom system prompt (overrides agent_type)"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override"
                },
                "reasoning_effort": {
                    "type": "string",
                    "description": "Optional reasoning effort (low/medium/high)"
                },
                "fork_history": {
                    "type": "string",
                    "description": "Optional history: 'none' (default), 'all', or a number N"
                },
                "depth": {
                    "type": "integer",
                    "description": "Nesting depth: 1=direct child, 2=grandchild, etc. Default 1."
                }
            },
            "required": ["task_name", "message"]
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(&self, args: Self::Args, ctx: &ToolContext) -> AgentResult<Self::Output> {
        let system_prompt = args
            .system_prompt
            .or(args
                .agent_type
                .map(|t| format!("You are a {} specialist.", t)))
            .unwrap_or_else(|| args.message.clone());

        let depth = args.depth;
        let tool_count = 0;

        match self
            .runtime
            .spawn_child_with_history(
                &args.task_name,
                system_prompt,
                depth,
                tool_count,
                args.fork_history,
                &ctx.session_id,
            )
            .await
        {
            Ok(agent_path) => {
                let _ = self
                    .runtime
                    .send_task(&agent_path, args.message.clone(), true);
                Ok(SpawnAgentOutput {
                    agent_path,
                    message: "Agent spawned successfully".to_string(),
                })
            }
            Err(e) => Ok(SpawnAgentOutput {
                agent_path: String::new(),
                message: format!("Failed to spawn agent: {}", e),
            }),
        }
    }
}
