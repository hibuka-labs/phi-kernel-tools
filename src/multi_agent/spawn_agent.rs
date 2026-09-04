use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::MultiAgentRuntime;
use serde::{Deserialize, Serialize};

/// Static system prompt for every spawned sub-agent.
///
/// Session 20260903_d8fc41dc: the previous design ran a Focus LLM call here
/// to expand the task into a bespoke system prompt. With a slow reasoning
/// model the 10 s budget failed 4/4, each spawn serialized a 10 s dead wait
/// into the parent's turn, and the children ran on the fallback template
/// anyway — producing excellent reports. Conclusion: a capable child only
/// needs its role stated plus a complete task; it plans by itself. Zero LLM
/// cost, zero truncation/timeout surface on the spawn path.
///
/// The last paragraph is generic path discipline (no domain assumptions):
/// children share the parent's process working directory, and session
/// 20260904_3eeb5610 showed a child silently resolving relative paths
/// against it while analyzing a different directory from its task — then
/// rationalizing the mismatch instead of re-checking. The concrete working
/// directory is appended per-spawn in [`SpawnAgentTool::call_typed`].
const CHILD_SYSTEM_PROMPT: &str = "\
You are a focused sub-agent spawned to handle exactly one task. The task is \
the first message you receive. Work autonomously — there is no one to ask \
mid-task: use the available tools to gather what you need, then deliver. \
Your final message is your deliverable: make it complete, structured, and \
self-contained, with concrete evidence (file paths, line references, \
measurements) for every claim. State limitations explicitly instead of \
guessing.

Path discipline: relative paths in tool calls resolve against your working \
directory (stated below). When the task specifies an absolute path, use it \
verbatim in tool calls. When what you observe contradicts what the task \
describes, re-verify your location and paths before drawing conclusions.";

/// Tool description, hoisted into a const so tests can guard its semantics
/// (the same drift class that put "keep `task` one sentence" here while the
/// expansion step it depended on no longer exists).
const DESCRIPTION: &str = "\
Spawn an independent sub-agent to handle a task.\n\
Sub-agents start with NO context of this conversation and see\n\
nothing but `task` — it must be COMPLETE and self-contained:\n\
the goal, the full paths of anything to analyze, the scope,\n\
and what the final report should cover (e.g. \"Analyze\n\
/Users/me/demo/codex: map the module structure, then explain\n\
the agent loop, tool system, and error-handling patterns;\n\
report with file paths and line references\").\n\
Give it a short unique name (task_name).\n\
Omit `model` unless the user explicitly asked for a different one.";

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SpawnAgentArgs {
    /// Unique name for this sub-agent (used in the agent path)
    pub task_name: String,
    /// What you want the sub-agent to do. The task must be COMPLETE and
    /// self-contained — the sub-agent starts with NO context of this
    /// conversation (unless fork_turns is set) and sees nothing but this
    /// text: include the goal, the full paths of anything to analyze, the
    /// scope, and what the final report should cover.
    pub task: String,
    /// How much of this conversation's history to give the sub-agent:
    /// `none` (default), `all`, or a number N for the last N turns.
    /// Use `none` only when the task is fully described in `task`.
    #[serde(default)]
    pub fork_turns: Option<String>,
    /// Model override for the sub-agent. Omit to inherit the parent's model.
    /// TODO(layer-3): request-level model routing is not wired yet — the
    /// value is accepted and stored on the child config but currently
    /// ignored at LLM-call time. Remove this note once llm-trait carries
    /// a per-request model field.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpawnAgentOutput {
    pub agent_path: String,
    pub message: String,
}

pub struct SpawnAgentTool {
    runtime: Arc<MultiAgentRuntime>,
    /// Directory the child's file tools resolve relative paths against.
    /// Injected as a fact into the child's system prompt — children share
    /// the parent's process cwd, which session 20260904_3eeb5610 showed a
    /// child silently analyzing the wrong directory from its task.
    workspace_root: std::path::PathBuf,
}

impl SpawnAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>, workspace_root: std::path::PathBuf) -> Self {
        Self {
            runtime,
            workspace_root,
        }
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
        DESCRIPTION
    }

    async fn call_typed(&self, args: Self::Args, ctx: &ToolContext) -> AgentResult<Self::Output> {
        // The task doubles as the message sent to the child after spawn.
        let message = args.task.clone();

        // Static role prompt + path discipline (see CHILD_SYSTEM_PROMPT),
        // with the concrete working directory as a plain fact — no LLM call
        // on the spawn path.
        let system_prompt = format!(
            "{CHILD_SYSTEM_PROMPT}\n\nWorking directory: {}",
            self.workspace_root.display()
        );

        // Non-LLM knob: permission comes from configuration
        // (ChildPermissionMode), not from the model. Depth is structurally
        // fixed at 1 (nesting absent, agent-works K5).
        let full_permission = false;

        // Fork-history priority: explicit LLM choice > configured default
        // (MultiAgentConfig::child_fork_history) > none.
        let fork_turns = args
            .fork_turns
            .or_else(|| self.runtime.child_fork_history().map(str::to_owned));

        match self
            .runtime
            .spawn_child_with_history(
                &args.task_name,
                system_prompt,
                full_permission,
                fork_turns,
                args.model,
                &ctx.session_id,
            )
            .await
        {
            Ok(agent_path) => {
                let _ = self
                    .runtime
                    .send_task(&agent_path, message, true);
                // TODO(layer-3): args.model is accepted but inert until
                // request-level model routing lands (see SpawnAgentArgs::model).
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

#[cfg(test)]
mod spawn_prompt_guard_tests {
    //! Guards the spawn-path semantics after the Focus-expansion removal
    //! (session 20260903_d8fc41dc): the child gets a static role prompt and
    //! a self-contained task — no LLM call, and no stale "keep `task` one
    //! sentence" guidance that only made sense when Focus expanded it.
    //!
    //! The path-discipline assertions guard session 20260904_3eeb5610: a
    //! child analyzed the wrong directory because it resolved relative paths
    //! against its (parent-inherited) working directory and rationalized the
    //! mismatch instead of re-checking.

    use super::{CHILD_SYSTEM_PROMPT, DESCRIPTION};

    #[test]
    fn child_prompt_states_autonomy_and_deliverable() {
        assert!(
            CHILD_SYSTEM_PROMPT.contains("Work autonomously"),
            "child prompt must tell the child it plans and works alone"
        );
        assert!(
            CHILD_SYSTEM_PROMPT.contains("final message is your deliverable"),
            "child prompt must define the final message as the report — \
             the parent only ever receives that text"
        );
    }

    #[test]
    fn child_prompt_carries_path_discipline() {
        assert!(
            CHILD_SYSTEM_PROMPT.contains("relative paths in tool calls resolve against"),
            "child prompt must state how relative paths resolve — children \
             inherit the parent cwd and cannot discover this fact themselves"
        );
        assert!(
            CHILD_SYSTEM_PROMPT.contains("use it verbatim in tool calls"),
            "task absolute paths must be mandated verbatim"
        );
        assert!(
            CHILD_SYSTEM_PROMPT.contains("re-verify your location and paths before drawing conclusions"),
            "observation/task contradictions must trigger a path re-check, \
             not a rationalization (session 20260904_3eeb5610)"
        );
        // Domain neutrality: the discipline must not assume what the task
        // is about (no "project"/"target directory" phrasing).
        assert!(
            !CHILD_SYSTEM_PROMPT.contains("project") && !CHILD_SYSTEM_PROMPT.contains("target"),
            "path discipline must stay domain-neutral"
        );
    }

    #[test]
    fn description_demands_self_contained_task() {
        assert!(
            DESCRIPTION.contains("COMPLETE and self-contained"),
            "task must be described as complete and self-contained"
        );
        assert!(
            DESCRIPTION.contains("NO context"),
            "description must warn that the child sees nothing but the task"
        );
        assert!(
            !DESCRIPTION.contains("one\nsentence") && !DESCRIPTION.contains("one sentence"),
            "the one-sentence guidance belonged to the removed Focus expansion"
        );
    }
}
