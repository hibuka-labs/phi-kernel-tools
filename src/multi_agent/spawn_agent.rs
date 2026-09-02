use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::focus::Focus;
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

/// System prompt template for Focus-based prompt generation.
///
/// When the LLM provides a short `task` description instead of a full
/// `system_prompt`, a Focus call expands it into a proper system prompt
/// using this template.
const FOCUS_SYSTEM_PROMPT: &str = "\
You are a prompt engineer. Given a short task description, generate a complete \
system prompt for a sub-agent. The system prompt should:
1. Define the agent's role clearly
2. State the specific task to accomplish
3. Mention available tools briefly if relevant
4. Be concise (under 500 words)

Output ONLY the system prompt text, nothing else.";

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SpawnAgentArgs {
    /// Unique name for this sub-agent (used in agent path)
    pub task_name: String,
    /// Initial task description for the sub-agent
    pub message: String,
    /// What you want the sub-agent to do. Describe the goal, not the steps.
    /// The sub-agent has its own tools and will figure out the approach.
    /// Example: "analyze the codex project structure and output a structured report"
    #[serde(default)]
    pub task: Option<String>,
    /// Optional role type that maps to a preset configuration
    #[serde(default)]
    pub agent_type: Option<String>,
    /// Optional custom system prompt (overrides task and agent_type).
    /// Prefer using `task` instead — it's shorter and less likely to be truncated.
    #[serde(default)]
    pub system_prompt: Option<String>,
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
        "Spawn an independent sub-agent to handle a task.\n\
         The sub-agent has its own tools (repo_map, read_file, list_files, etc.)\n\
         and can reason, read files, and explore codebases autonomously.\n\
         Use the `task` field to describe what you want done (e.g. \"analyze the\n\
         codex project structure and output a structured report\"). The system\n\
         generates a full prompt automatically — no need to write detailed\n\
         instructions. The sub-agent will figure out the steps itself."
    }

    async fn call_typed(&self, args: Self::Args, ctx: &ToolContext) -> AgentResult<Self::Output> {
        // Priority: system_prompt > task (via Focus) > agent_type > message
        let system_prompt = if let Some(sp) = args.system_prompt {
            // Explicit system prompt — use as-is
            sp
        } else if let Some(task) = args.task {
            // Short task description — expand via Focus
            expand_task_via_focus(self.runtime.client(), &task).await
        } else if let Some(agent_type) = args.agent_type {
            // Preset role type
            format!("You are a {} specialist.", agent_type)
        } else {
            // Fallback: use message as system prompt (legacy behavior)
            args.message.clone()
        };

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

/// Expand a short task description into a full system prompt via Focus.
///
/// This is the core of the "task-based spawn" approach: the LLM writes a
/// short task (e.g. "analyze codex structure"), and Focus generates a
/// proper system prompt from it, avoiding the truncation issue that occurs
/// when the LLM tries to inline a full prompt in tool call arguments.
async fn expand_task_via_focus(
    client: &Arc<dyn agent_base::llm_trait::LlmProvider>,
    task: &str,
) -> String {
    let focus = Focus::new(Arc::clone(client), FOCUS_SYSTEM_PROMPT);
    let timeout = std::time::Duration::from_secs(10);

    // Focus forces JSON output, so we wrap the prompt in a struct.
    #[derive(serde::Deserialize)]
    struct PromptResult {
        prompt: String,
    }

    // Ask Focus to return {"prompt": "..."} format.
    let input = format!(
        "Generate a system prompt for a sub-agent whose task is: {}\n\
         Return JSON: {{\"prompt\": \"<the system prompt>\"}}",
        task
    );

    match focus.ask::<PromptResult>(&input, timeout).await {
        Ok(output) => {
            tracing::info!(
                task = task,
                prompt_len = output.result.prompt.len(),
                "Focus expanded task into system prompt"
            );
            output.result.prompt
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                task = task,
                "Focus failed to expand task, using fallback template"
            );
            // Fallback: simple template
            format!(
                "You are a helpful assistant. Your task is: {}. \
                 Use the available tools to complete this task effectively.",
                task
            )
        }
    }
}
