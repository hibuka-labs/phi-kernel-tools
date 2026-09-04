use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListAgentsArgs {}

#[derive(Debug, Serialize)]
pub struct ListAgentItem {
    pub agent_path: String,
    pub status: String,
    /// Tool calls the agent has actually executed (monotonic; grows while
    /// it works). A frozen count with a stale `last_activity_secs` — not a
    /// low count — is the stall signal.
    pub tool_calls: usize,
    /// Seconds since the agent's last activity (task start or tool call);
    /// absent until the agent receives its first task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_activity_secs: Option<u64>,
    /// First-line excerpt of what the agent was asked to do — just enough to
    /// tell agents apart. The full task text was the spawner's own input, so
    /// re-sending it on every poll only burns prompt tokens (session
    /// 20260903_9255c25e: 65 polls × 4 full tasks ≈ 160KB of context).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
}

/// Task excerpt length in characters.
const TASK_EXCERPT_CHARS: usize = 60;

/// First line of the task, truncated to [`TASK_EXCERPT_CHARS`] chars.
fn task_excerpt(task: &str) -> String {
    let first_line = task.lines().next().unwrap_or("");
    let mut s: String = first_line.chars().take(TASK_EXCERPT_CHARS).collect();
    if first_line.chars().count() > TASK_EXCERPT_CHARS {
        s.push('…');
    }
    s
}

pub struct ListAgentsTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl ListAgentsTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for ListAgentsTool {
    type Args = ListAgentsArgs;
    type Output = Vec<ListAgentItem>;

    fn name(&self) -> &'static str {
        "list_agents"
    }

    fn description(&self) -> &'static str {
        "List all active sub-agents, their status, and a short task excerpt.\n\
         Spot-check tool ONLY: call it at most once to see what each agent is\n\
         doing. NEVER poll it and NEVER use it to wait — when every sub-agent\n\
         has finished, their full reports are pushed to you automatically in\n\
         one message. A single snapshot is not evidence of a stall, and a\n\
         `done` status means the result is already en route to you. Repeated\n\
         calls are pure token burn and change nothing.\n\
         Status: idle (ready), running (executing), done (completed, result\n\
         pending delivery)."
    }

    async fn call_typed(&self, _args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        let agents = self.runtime.list_agents();
        Ok(agents
            .into_iter()
            .map(|a| ListAgentItem {
                agent_path: a.agent_path,
                status: a.status,
                tool_calls: a.tool_calls,
                last_activity_secs: a.last_activity_secs,
                task: a.task.as_deref().map(task_excerpt),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::task_excerpt;

    #[test]
    fn excerpt_truncates_long_first_line_with_ellipsis() {
        let task = "分析 /Users/kangzengchen/source/buka/demo/codex 工程的完整结构和核心设计，重点关注架构";
        let out = task_excerpt(task);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 61); // 60 chars + ellipsis
        assert!(!out.contains('\n'));
    }

    #[test]
    fn excerpt_keeps_short_task_verbatim_without_ellipsis() {
        assert_eq!(task_excerpt("短任务"), "短任务");
    }

    #[test]
    fn excerpt_uses_first_line_only() {
        let out = task_excerpt("第一行\n第二行不该出现");
        assert_eq!(out, "第一行");
    }
}
