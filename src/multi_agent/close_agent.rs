use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CloseAgentArgs {
    /// Agent path to close (e.g., 'root/searcher')
    pub agent_path: String,
}

#[derive(Debug, Serialize)]
pub struct CloseAgentOutput {
    pub closed: bool,
    pub previous_status: String,
    pub message: String,
}

/// Pre-close warnings, computed from the facts `list_agents` already
/// exposes. Session 20260904_841ed65b: the parent closed four *healthy*
/// done agents; the delivery gap (`pending_results`) makes that shape
/// detectable, and the warning names what closing does and does not do.
fn close_warnings(status: &str, pending_results: usize) -> Vec<String> {
    let mut warnings = Vec::new();
    if pending_results > 0 {
        warnings.push(format!(
            "warning: this agent has {pending_results} report(s) not yet delivered \
             to you — they stay in the batch and WILL still arrive; closing does \
             not revoke them."
        ));
    }
    if status == "queued" {
        warnings.push(
            "warning: this agent has a queued task that is dropped now and will \
             never run."
                .to_string(),
        );
    }
    warnings
}

pub struct CloseAgentTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl CloseAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for CloseAgentTool {
    type Args = CloseAgentArgs;
    type Output = CloseAgentOutput;

    fn name(&self) -> &'static str {
        "close_agent"
    }

    fn description(&self) -> &'static str {
        "Close a sub-agent and release its resources. The cancel signal is\n\
         set: the agent finishes its current task at the task boundary and\n\
         then exits; tasks still queued are dropped. Reports it already\n\
         delivered remain in the batch and still reach you — closing never\n\
         revokes them. Use ONLY when an agent is truly stuck or its task is\n\
         obsolete; do NOT close running agents to 'clean up' — waiting is\n\
         passive and the batch wakes you with everything."
    }

    async fn call_typed(&self, args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        // Facts read BEFORE closing: after close the agent is unregistered
        // and gone from the listing.
        let pre = self
            .runtime
            .list_agents()
            .into_iter()
            .find(|a| a.agent_path == args.agent_path);
        let warnings = pre
            .as_ref()
            .map(|a| close_warnings(&a.status, a.pending_results))
            .unwrap_or_default();

        match self.runtime.close_agent(&args.agent_path) {
            Ok(result) => {
                let mut message = result.message;
                for w in &warnings {
                    message.push(' ');
                    message.push_str(w);
                }
                Ok(CloseAgentOutput {
                    closed: result.closed,
                    previous_status: result.previous_status,
                    message,
                })
            }
            Err(e) => Ok(CloseAgentOutput {
                closed: false,
                previous_status: pre.map(|a| a.status).unwrap_or_else(|| "unknown".into()),
                message: e,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::close_warnings;

    #[test]
    fn pending_reports_warn_that_delivery_survives_close() {
        let w = close_warnings("done", 2);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("2 report(s)"));
        assert!(w[0].contains("WILL still arrive"));
        assert!(w[0].contains("does not revoke"));
    }

    #[test]
    fn queued_task_warns_that_it_is_dropped() {
        let w = close_warnings("queued", 0);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("queued task"));
        assert!(w[0].contains("never run"));
    }

    #[test]
    fn running_with_no_pending_gets_no_warning() {
        // Closing a genuinely stuck running agent is the legitimate case —
        // it must not be nagged.
        assert!(close_warnings("running", 0).is_empty());
        assert!(close_warnings("done", 0).is_empty());
    }

    #[test]
    fn queued_and_pending_warn_together() {
        let w = close_warnings("queued", 1);
        assert_eq!(w.len(), 2, "{w:?}");
    }
}
