use std::sync::Arc;

use agent_base::{AgentResult, ToolContext, TypedTool};
use agent_works::multi_agent::ChildToolCapability;
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
/// Session 20260904_efad759c forced the second half of the synthesis: the
/// parent was asked to write each task as a complete self-contained brief,
/// and 4 briefs in one response exhausted the model's output budget — the
/// truncated spawn_agent call never executed. So the brief-writing duty
/// moved here: the parent's `task` stays short (3-5 sentences) and the
/// standing report format below is appended free of charge. The task
/// carries only what is per-spawn (goal, paths, scope, focus); everything
/// every spawn shares lives in this prompt.
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
Your final message is your deliverable — the parent receives nothing else.

Report format (applies unless the task specifies its own): structure the \
final message as (1) Findings — the direct answer to the task, every claim \
backed by concrete evidence (file paths, line references, measurements); \
(2) Method — what you examined and what you deliberately skipped; \
(3) Limitations — what you could not verify or complete. If the task does \
not bound the scope, examine only what the goal requires, and record that \
choice under Method.

Path discipline: relative paths in tool calls resolve against your working \
directory (stated below). When the task specifies an absolute path, use it \
verbatim in tool calls. When what you observe contradicts what the task \
describes, re-verify your location and paths before drawing conclusions.";

/// Tool description, hoisted into a const so tests can guard its semantics.
///
/// Session 20260904_efad759c: "COMPLETE and self-contained" read as license
/// for essay-length briefs; 4 of them in one response truncated the spawn
/// call into empty args. The guidance now caps the task at 3-5 sentences —
/// args length is the failure surface of this call — and points at the
/// automatic role prompt + report format instead.
const DESCRIPTION: &str = "\
Spawn an independent sub-agent to handle a task.\n\
A sub-agent costs more than it looks: it knows only what `task` says, you see\n\
only what its report says, both handoffs drop detail, and its mistakes come\n\
back in the same confident register as its findings. Do the work yourself\n\
when it is a handful of tool calls or a lookup whose target you already know\n\
— a check you could run inline is not worth an agent. Reach for this when\n\
the parts are independent and can run at once, or when answering would mean\n\
reading across many files and you want the conclusion, not the file dumps.\n\
When in doubt, don't spawn.\n\
Sub-agents start with NO context of this conversation and see\n\
nothing but `task` (a standard role prompt and report format are\n\
added automatically). Keep `task` SHORT — 3-5 sentences covering:\n\
the goal, the full paths of anything to analyze, the scope, and\n\
what the report should focus on.\n\
Brief it like the peer it is: say what you already ruled out so it\n\
does not go and rule it out again, and point it at the files and\n\
docs worth reading instead of retyping what you already read.\n\
Handing over your own hypothesis tends to get that hypothesis\n\
confirmed back — say what you suspect and what would falsify it.\n\
Do NOT write a long detailed brief — extra prose only inflates this\n\
tool call, which is its main truncation failure mode.\n\
Give it a short unique name in `task_name` — a label like\n\
`runbook-rules`, NOT the task itself. The work goes in `task`:\n\
`task_name` names the agent, `task` is what it does. Mixing them up\n\
fails the call outright, with nothing done.\n\
Spawning does not block you: keep working and check on it later with\n\
`list_agents` (status, elapsed time, tool calls). A sub-agent runs until\n\
its report is ready; if one looks stuck, `close_agent` stops it.\n\
Omit `tools` for read-only research tasks (the default).\n\
Request \"write\" only when the task must edit files or run\n\
commands; use a preset name (researcher/coder/reviewer/tester)\n\
for its standard persona. In ask mode every sub-agent write is\n\
confirmed by the user (the popup names the sub-agent).\n\
Its report is not shown to the user — relay what matters.\n\
Omit `model` unless the user explicitly asked for a different one.";

/// Legal values for the `tools` parameter (schema enum constraint; an unknown
/// value = tool-call error, and the LLM can retry with a legal one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolsSpec {
    /// Read-only research (the default when `tools` is omitted).
    ReadOnly,
    /// Tasks that edit files / run commands. In ask mode the user confirms each one.
    Write,
    /// Code research (read-only preset).
    Researcher,
    /// Code writing preset (includes write tools).
    Coder,
    /// Code review (read-only preset).
    Reviewer,
    /// Test writing & execution preset (includes write tools).
    Tester,
}

impl ToolsSpec {
    /// Maps to the framework capability vocabulary (D2).
    pub fn capability(self) -> ChildToolCapability {
        match self {
            Self::ReadOnly => ChildToolCapability::ReadOnly,
            Self::Write => ChildToolCapability::Write,
            Self::Researcher => ChildToolCapability::Preset("researcher"),
            Self::Coder => ChildToolCapability::Preset("coder"),
            Self::Reviewer => ChildToolCapability::Preset("reviewer"),
            Self::Tester => ChildToolCapability::Preset("tester"),
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SpawnAgentArgs {
    /// Label for this sub-agent (used in the agent path) — a short handle like
    /// `runbook-rules`. NOT the task: the work belongs in `task`. This names
    /// the agent, it does not say what the agent does.
    pub task_name: String,
    /// What you want the sub-agent to do — SHORT, 3-5 sentences: the goal,
    /// the full paths of anything to analyze, the scope, and what the
    /// report should focus on. The sub-agent starts with NO context of
    /// this conversation (unless fork_turns is set) and sees nothing but
    /// this text; a standard role prompt and report format are added
    /// automatically, so do not spell out report requirements in detail.
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
    /// Tool face: omit = read_only. Omit it for read-only research tasks; request
    /// "write" for tasks that need to edit files / run commands; for standard
    /// personas use the preset name directly (researcher/coder/reviewer/tester).
    /// In ask mode every sub-agent write pops up for user confirmation.
    #[serde(default)]
    pub tools: Option<ToolsSpec>,
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

        // D2: the schema defaults to read_only — an LLM-facing spawn always
        // requests a capability explicitly; the None semantics of the legacy
        // programmatic path never comes through here.
        let requested = args
            .tools
            .map(|t| t.capability())
            .unwrap_or(ChildToolCapability::ReadOnly);
        let capability = Some(requested);

        match self
            .runtime
            .spawn_child_with_history(
                &args.task_name,
                system_prompt,
                full_permission,
                fork_turns,
                args.model,
                capability,
                &ctx.session_id,
            )
            .await
        {
            Ok(echo) => {
                // The initial task IS the spawn's purpose. A failed delivery
                // must not be swallowed: the child would sit registered with
                // zero deliveries, which permanently blocks the runtime's
                // quiescence (the fan-in batch can never fire), and the
                // parent would wait forever for a result that was promised.
                // Close the orphan and report the truth instead.
                match self.runtime.send_task(&echo.agent_path, message, true) {
                    Ok(true) => {
                        // TODO(layer-3): args.model is accepted but inert
                        // until request-level model routing lands (see
                        // SpawnAgentArgs::model).
                        // D3/finding 9: echo the actual capability — spell out
                        // any degradation (avoid "fake success"), and keep the
                        // default read-only output clean.
                        let mut msg = "Agent spawned successfully".to_string();
                        if echo.recycled {
                            // The same-path predecessor was already terminal
                            // (done/closed); this spawn took over its
                            // registration slot — the parent agent must see
                            // the fact, never a silent takeover (the three
                            // spawn collisions of session
                            // 20260920_5ba1bed4).
                            msg.push_str(" (recycled a finished agent with the same path)");
                        }
                        if let Some(why) = &echo.degraded_reason {
                            msg.push_str(&format!(" (tools degraded to read-only: {why})"));
                        } else {
                            let registered: Vec<&str> =
                                echo.registered_tools.iter().map(String::as_str).collect();
                            msg.push_str(&format!(
                                " (tools: {}; registered: {})",
                                requested.label(),
                                registered.join(", ")
                            ));
                        }
                        Ok(SpawnAgentOutput {
                            agent_path: echo.agent_path,
                            message: msg,
                        })
                    }
                    other => {
                        // Ok(false): the mailbox vanished between spawn and
                        // send (the child task loop already exited). Err: the
                        // agent path could not be resolved. Surface the real
                        // reason instead of a bare "delivery failed", which
                        // used to hide a malformed name behind a generic error.
                        let reason = match other {
                            Ok(_) => "agent is no longer registered".to_string(),
                            Err(e) => e,
                        };
                        // Best effort: a close failure here means the
                        // runtime is tearing down anyway; the caller cannot
                        // use the agent either way.
                        let close_err = self.runtime.close_agent(&echo.agent_path).err();
                        let why = close_err
                            .map(|e| format!(" (cleanup also failed: {})", e))
                            .unwrap_or_default();
                        Ok(SpawnAgentOutput {
                            agent_path: String::new(),
                            message: format!(
                                "Agent spawned but task delivery failed ({reason}), \
                                 agent closed. Retry spawn if needed{}",
                                why
                            ),
                        })
                    }
                }
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
    //! a short task — no LLM call. Session 20260904_efad759c settled the
    //! task-length side: "COMPLETE and self-contained" briefs truncated the
    //! spawn call itself, so the parent writes 3-5 sentences and the static
    //! report format lives in CHILD_SYSTEM_PROMPT.
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
            CHILD_SYSTEM_PROMPT
                .contains("re-verify your location and paths before drawing conclusions"),
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
    fn child_prompt_carries_report_scaffold() {
        // Session 20260904_efad759c: report structure used to be spelled out
        // inside each parent-written task — moving it here is what lets the
        // task stay at 3-5 sentences. All three sections must be present,
        // plus the scope fallback for unbounded tasks.
        for section in ["Findings", "Method", "Limitations"] {
            assert!(
                CHILD_SYSTEM_PROMPT.contains(section),
                "report format must define the {section} section"
            );
        }
        assert!(
            CHILD_SYSTEM_PROMPT.contains("If the task does not bound the scope"),
            "report format must carry the scope fallback — short tasks \
             routinely omit explicit scope bounds"
        );
    }

    /// Value: protects=the decision to delegate, not just how to delegate —
    /// a tool that teaches briefing quality and says nothing about cost
    /// invites one where an inline check would do.
    /// why_new=the description capped the brief and taught its shape, but
    /// never said when the tool is the wrong one.
    /// seam=none; no_claim=that the model obeys it.
    #[test]
    fn description_says_when_not_to_spawn() {
        assert!(
            DESCRIPTION.contains("When in doubt"),
            "a delegation tool must carry its own brake: {DESCRIPTION}"
        );
        assert!(
            DESCRIPTION.contains("costs more than it looks"),
            "and name the cost — the caller cannot see what it paid: {DESCRIPTION}"
        );
        assert!(
            DESCRIPTION.contains("inline"),
            "a check you could run yourself is not worth a sub-agent: {DESCRIPTION}"
        );
    }

    /// Value: protects=the brief the child gets is a brief a peer could act
    /// on, not a topic; fails_when=length is capped but quality is not, and
    /// the child re-derives what the parent already knew.
    /// why_new=the description capped `task` at 3-5 sentences and said nothing
    /// about what those sentences must carry.
    /// seam=none; no_claim=brief quality at runtime.
    #[test]
    fn description_says_how_to_write_the_brief() {
        assert!(
            DESCRIPTION.contains("ruled out"),
            "a brief must carry what the parent already ruled out, or the \
             child re-investigates it: {DESCRIPTION}"
        );
        assert!(
            DESCRIPTION.contains("hypothesis"),
            "handing over a hypothesis invites it being confirmed back — the \
             brief must say so: {DESCRIPTION}"
        );
    }

    #[test]
    fn description_demands_short_task() {
        assert!(
            DESCRIPTION.contains("3-5 sentences"),
            "task length must be capped explicitly — args length is the \
             truncation failure surface of this call (20260904_efad759c)"
        );
        assert!(
            DESCRIPTION.contains("NO context"),
            "description must warn that the child sees nothing but the task"
        );
        assert!(
            DESCRIPTION.contains("added automatically"),
            "description must point at the automatic role prompt + report \
             format so the model does not spell them out in `task`"
        );
        assert!(
            DESCRIPTION.contains("Do NOT write a long detailed brief"),
            "the essay-length failure mode must be named"
        );
        // The old regime's key phrase must not come back.
        assert!(
            !DESCRIPTION.contains("COMPLETE and self-contained"),
            "self-contained-brief guidance is the truncation surface this \
             description was rewritten to remove"
        );
    }

    /// Value: protects=the two-field split — `task_name` labels the agent and
    /// `task` is the work, and the names invite the opposite reading. Session
    /// 20261115_08a9ccae put the whole brief in `task_name` and the call died
    /// with "missing field `task`", twice, before it got it right.
    /// fails_when=the description stops saying which field is which.
    /// why_new=it only said "give it a short unique name", which reads as "name
    /// the task". seam=description + schema field doc; no_claim=that the field
    /// name is a good one — renaming it is the better fix and is still open.
    #[test]
    fn description_separates_the_label_from_the_work() {
        assert!(
            DESCRIPTION.contains("NOT the task"),
            "must say what `task_name` is not — that is the error the model made"
        );
        assert!(
            DESCRIPTION.contains("The work goes in `task`"),
            "and must say where the work goes"
        );
        assert!(
            DESCRIPTION.contains("fails the call outright"),
            "the cost of mixing them up must be named, or it reads as a hint"
        );
    }
}

#[cfg(test)]
mod tools_spec_tests {
    use super::{DESCRIPTION, SpawnAgentArgs, ToolsSpec};
    use agent_works::multi_agent::ChildToolCapability;

    #[test]
    fn tools_spec_maps_to_capabilities() {
        assert_eq!(
            ToolsSpec::ReadOnly.capability(),
            ChildToolCapability::ReadOnly
        );
        assert_eq!(ToolsSpec::Write.capability(), ChildToolCapability::Write);
        assert_eq!(
            ToolsSpec::Coder.capability(),
            ChildToolCapability::Preset("coder")
        );
        assert_eq!(
            ToolsSpec::Tester.capability(),
            ChildToolCapability::Preset("tester")
        );
    }

    #[test]
    fn tools_is_optional_and_defaults_to_read_only() {
        // No `tools` arg = read_only (serde default). Parsing goes through serde:
        let json = r#"{"task_name":"n","task":"t"}"#;
        let args: SpawnAgentArgs = serde_json::from_str(json).unwrap();
        assert!(args.tools.is_none());
        assert_eq!(
            args.tools.map(|t| t.capability()),
            None,
            "omitting = no capability-layer request; the spawn tool layer maps it to ReadOnly (D2 default column)"
        );
    }

    #[test]
    fn tools_accepts_all_enum_values() {
        for (raw, expected) in [
            ("read_only", ChildToolCapability::ReadOnly),
            ("write", ChildToolCapability::Write),
            ("researcher", ChildToolCapability::Preset("researcher")),
            ("coder", ChildToolCapability::Preset("coder")),
            ("reviewer", ChildToolCapability::Preset("reviewer")),
            ("tester", ChildToolCapability::Preset("tester")),
        ] {
            let json = format!(r#"{{"task_name":"n","task":"t","tools":"{raw}"}}"#);
            let args: SpawnAgentArgs = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("{raw} must deserialize: {e}"));
            assert_eq!(args.tools.unwrap().capability(), expected);
        }
    }

    #[test]
    fn illegal_tools_value_is_an_error() {
        // Unknown name = tool-call error (the LLM retries with a legal value) —
        // serde rejecting it IS the error path.
        let json = r#"{"task_name":"n","task":"t","tools":"translator"}"#;
        assert!(serde_json::from_str::<SpawnAgentArgs>(json).is_err());
    }

    #[test]
    fn description_guides_capability_choice() {
        assert!(
            DESCRIPTION.contains("read-only"),
            "description must say read-only tasks omit tools"
        );
        assert!(
            DESCRIPTION.contains("\"write\""),
            "description must name the write value"
        );
        assert!(
            DESCRIPTION.contains("preset"),
            "description must mention preset names"
        );
    }
}
