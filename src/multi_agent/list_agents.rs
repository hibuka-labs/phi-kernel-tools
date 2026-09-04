use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListAgentsArgs {}

#[derive(Debug, Serialize)]
pub struct ListAgentsOutput {
    pub agents: Vec<ListAgentItem>,
    /// Delivery facts the parent cannot otherwise verify. Session
    /// 20260904_841ed65b: a mid-turn parent saw `done` children but no
    /// reports in its context and concluded the system failed to deliver —
    /// the reports were held for the batch. This note turns that blind
    /// trust into a checkable fact. `None` while anyone is still working.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivery_note: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListAgentItem {
    pub agent_path: String,
    pub status: String,
    /// Tool calls the agent has actually executed (monotonic; grows while
    /// it works). A frozen count with a stale `last_activity_secs` — not a
    /// low count — is the stall signal.
    pub tool_calls: usize,
    /// Seconds spent in the current task; present only while `running`.
    /// Feeds the framework's stall reaper, and lets the parent read "how
    /// long has it been at this" without polling twice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub running_secs: Option<u64>,
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
    /// Reports this agent has posted that have not reached you yet — they
    /// are held for the next batch. Zero means nothing is in flight from
    /// this agent. A `done` agent with `pending_results > 0` needs nothing
    /// from you: its report IS en route, ending your turn is what delivers it.
    #[serde(skip_serializing_if = "is_zero")]
    pub pending_results: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
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
    type Output = ListAgentsOutput;

    fn name(&self) -> &'static str {
        "list_agents"
    }

    fn description(&self) -> &'static str {
        "List all active sub-agents, their status, and a short task excerpt.\n\
         Spot-check tool ONLY: call it at most once to see what each agent is\n\
         doing. NEVER poll it and NEVER use it to wait — when every sub-agent\n\
         has finished, their full reports are pushed to you automatically in\n\
         one message at the END of your turn. A single snapshot is not\n\
         evidence of a stall. `pending_results` is the delivery fact: it\n\
         counts reports that exist but are still held for the batch. A\n\
         `done` agent with `pending_results > 0` is completely healthy —\n\
         its report IS en route, and ending your turn is what delivers it.\n\
         Repeated calls are pure token burn and change nothing.\n\
         Status: queued (task accepted, waiting in its queue), running\n\
         (executing; `running_secs` = seconds in the current task), done\n\
         (no work left — check `pending_results` for delivery)."
    }

    async fn call_typed(&self, _args: Self::Args, _ctx: &ToolContext) -> AgentResult<Self::Output> {
        let agents = self.runtime.list_agents();
        let total_pending: usize = agents.iter().map(|a| a.pending_results).sum();
        let busy = agents
            .iter()
            .any(|a| a.status == "running" || a.status == "queued");
        let delivery_note = delivery_note(total_pending, busy, agents.len());
        Ok(ListAgentsOutput {
            agents: agents
                .into_iter()
                .map(|a| ListAgentItem {
                    agent_path: a.agent_path,
                    status: a.status,
                    tool_calls: a.tool_calls,
                    running_secs: a.running_secs,
                    last_activity_secs: a.last_activity_secs,
                    task: a.task.as_deref().map(task_excerpt),
                    pending_results: a.pending_results,
                })
                .collect(),
            delivery_note,
        })
    }
}

/// Delivery facts for the mid-turn parent. Session 20260904_841ed65b: a
/// parent watching `done` statuses with no reports in context concluded
/// "the system didn't deliver" — the reports were batch-held by design.
/// These notes make the hold visible and name the one action that
/// releases it: ending the turn.
fn delivery_note(total_pending: usize, busy: bool, agent_count: usize) -> Option<String> {
    if agent_count == 0 || total_pending == 0 {
        return None;
    }
    if busy {
        Some(format!(
            "{total_pending} finished report(s) are held for the batch while the \
             remaining agent(s) work. Waiting is passive: end your turn and the \
             batch will wake you with everything. Do NOT poll and do NOT redo \
             their work."
        ))
    } else {
        Some(format!(
            "All {total_pending} finished report(s) are ready and held for the \
             batch. End your turn NOW to receive them. Do NOT poll and do NOT \
             redo their work."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{delivery_note, task_excerpt};

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

    #[test]
    fn note_is_silent_without_agents_or_without_pending() {
        assert_eq!(delivery_note(0, false, 0), None, "nothing spawned");
        assert_eq!(
            delivery_note(0, false, 3),
            None,
            "agents exist, nothing undelivered"
        );
        assert_eq!(
            delivery_note(0, true, 3),
            None,
            "still working, nothing finished yet"
        );
    }

    #[test]
    fn note_all_done_tells_the_parent_to_end_its_turn() {
        // The exact 841ed65b shape: everyone reads `done`, reports are held
        // for the batch, and the parent needs to be told delivery is normal
        // and that ending the turn releases it.
        let note = delivery_note(4, false, 4).expect("note expected");
        assert!(note.contains("4 finished report(s)"));
        assert!(note.contains("End your turn NOW"));
        assert!(note.contains("Do NOT poll"));
        assert!(note.contains("NOT redo"));
    }

    #[test]
    fn note_partial_delivery_says_waiting_is_passive() {
        let note = delivery_note(2, true, 4).expect("note expected");
        assert!(note.contains("2 finished report(s)"));
        assert!(note.contains("Waiting is passive"));
        assert!(note.contains("end your turn"));
    }
}
