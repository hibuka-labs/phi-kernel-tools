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
    pub tool_count: usize,
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
        "List all active sub-agents and their status.\n\
         Status: idle (ready), running (executing), done (completed, awaiting close or new task)."
    }

    async fn call_typed(&self, _args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        let agents = self.runtime.list_agents();
        Ok(agents
            .into_iter()
            .map(|a| ListAgentItem {
                agent_path: a.agent_path,
                status: a.status,
                tool_count: a.tool_count,
            })
            .collect())
    }
}
