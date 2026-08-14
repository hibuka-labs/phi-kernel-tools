use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SpawnAgentArgs {
    /// Unique name for this sub-agent (used in agent path)
    pub task_name: String,
    /// Initial task description for the sub-agent
    pub message: String,
    /// Optional role type that maps to a preset configuration
    #[serde(default)]
    pub agent_type: Option<String>,
    /// Optional custom system prompt (overrides agent_type)
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Optional model override
    #[serde(default)]
    pub model: Option<String>,
    /// Optional reasoning effort (low/medium/high)
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Optional history: 'none' (default), 'all', or a number N
    #[serde(default)]
    pub fork_history: Option<String>,
    /// Nesting depth for this agent. 1 = direct child of root, 2 = grandchild, etc.
    /// Defaults to 1 (direct child).
    #[serde(default = "default_depth")]
    pub depth: i32,
    /// Whether to grant the sub-agent full permission to run tools without
    /// approval. Default `false` = deny all approvals (safe). Set `true` only
    /// when the sub-agent is trusted to take dangerous actions.
    #[serde(default)]
    pub full_permission: bool,
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
         Use agent_type for preset roles, or system_prompt for custom instructions.\n\
         Set full_permission=true only when the sub-agent may take dangerous\n\
         actions without approval (default false = deny all)."
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
                args.full_permission,
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
