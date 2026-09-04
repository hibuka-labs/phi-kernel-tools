use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

/// Deprecated argument shim (§8.3, kept one version).
///
/// The legacy `interrupt` field is dropped from the schema: it was a dead
/// parameter (defect K2 — tasks always ran serially inside a child, the flag
/// changed nothing). Old JSON that still sends `"interrupt": ...` parses
/// fine; the field is ignored.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FollowupTaskArgs {
    /// Target agent path (e.g., 'root/searcher')
    pub agent_path: String,
    /// Task description for the sub-agent
    pub task: String,
}

#[derive(Debug, Serialize)]
pub struct FollowupTaskOutput {
    pub accepted: bool,
    pub agent_path: String,
}

/// Deprecated: use `send_message` with `trigger=true` instead.
///
/// Kept working for exactly one release (design doc §8.3): every call
/// forwards to the runtime's task-queue path (identical semantics to
/// `send_message(trigger=true)`) and logs a migration warning. It is no
/// longer registered by [`create_all_tools`](super::create_all_tools).
#[deprecated(note = "use send_message with trigger=true (design doc §8.3)")]
pub struct FollowupTaskTool {
    runtime: Arc<MultiAgentRuntime>,
}

#[allow(deprecated)]
impl FollowupTaskTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[allow(deprecated)]
#[async_trait::async_trait]
impl TypedTool for FollowupTaskTool {
    type Args = FollowupTaskArgs;
    type Output = FollowupTaskOutput;

    fn name(&self) -> &'static str {
        "followup_task"
    }

    fn description(&self) -> &'static str {
        "DEPRECATED — use send_message with trigger=true instead.\n\
         Send a task to a sub-agent and trigger execution. Tasks run serially\n\
         inside a child (the old `interrupt` flag was a no-op and is gone)."
    }

    async fn call_typed(&self, args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        tracing::warn!(
            "followup_task is deprecated — use send_message(trigger=true); \
             forwarding this call unchanged"
        );
        // Same path send_message(trigger=true) takes. The old shape wrapped
        // errors into the output (accepted=false, "error: ..." in
        // agent_path); that behaviour is frozen for this one version — a
        // forwarding shim must not change error semantics either.
        match self.runtime.send_task(&args.agent_path, args.task, false) {
            Ok(accepted) => Ok(FollowupTaskOutput {
                accepted,
                agent_path: args.agent_path,
            }),
            Err(e) => Ok(FollowupTaskOutput {
                accepted: false,
                agent_path: format!("error: {e}"),
            }),
        }
    }
}
