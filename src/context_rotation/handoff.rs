//! Mechanical handoff — the activity ledger (context rotation).
//!
//! Session 20260909_e7053736 proved the design rule "don't bet the handoff
//! on the LLM": 9/9 fallback nudges were ignored and the model re-read the
//! same files in all 10 windows. This module extracts the handoff
//! deterministically from the outgoing window's message list — zero model
//! participation:
//!
//! - **Covered actions**: every work tool call, mapped through a
//!   phimint-specific argument table (`read_file` → path+lines,
//!   `execute_command` → command head, ...). Context-management churn
//!   (`history_*`, `notes_read_*`) is excluded — it is not work.
//! - **Last narration**: the model's final non-empty assistant text, so the
//!   fresh window knows where the train stopped.
//!
//! The compactor injects the ledger into the fresh window via the existing
//! thread-hint slot and mirrors it to `notes/activity_ledger.md`, so even
//! at the smallest viable budget `notes_list_files` is never empty.

use agent_base::ChatMessage;

/// Keep at most this many distinct actions (the most recent ones).
const MAX_ENTRIES: usize = 30;
/// Per-entry char cap (paths/commands can be long; CJK-safe truncation).
const MAX_ENTRY_CHARS: usize = 80;
/// Last-narration char cap.
const MAX_NARRATION_CHARS: usize = 200;
/// Whole-ledger char cap (injected into every fresh window — must stay
/// small against a minimum viable budget of ~4k work-room tokens).
const MAX_LEDGER_CHARS: usize = 1200;

/// Keep at most this many artifact references (the most recent ones).
const MAX_INDEX_ENTRIES: usize = 10;
/// Whole-index char cap (same order as the ledger: tiny against a fresh
/// window's work room).
const MAX_INDEX_CHARS: usize = 2_000;

/// Extract the activity ledger from an outgoing window's messages.
///
/// Returns `None` when the window contains no work worth handing off (no
/// tool calls and no assistant narration — e.g. a window that died to its
/// seed).
pub fn extract_ledger(messages: &[ChatMessage]) -> Option<String> {
    let mut entries: Vec<String> = Vec::new();
    let mut narration: Option<String> = None;

    for msg in messages {
        if let ChatMessage::Assistant {
            content,
            tool_calls,
            ..
        } = msg
        {
            if let Some(text) = content {
                let flat = one_line(text);
                if !flat.is_empty() {
                    // Keep the LAST narration — where the train stopped.
                    narration = Some(truncate_chars(&flat, MAX_NARRATION_CHARS));
                }
            }
            if let Some(calls) = tool_calls {
                for call in calls {
                    if let Some(entry) = ledger_entry(&call.name, &call.arguments)
                        && !entries.contains(&entry)
                    {
                        entries.push(entry);
                    }
                }
            }
        }
    }

    if entries.is_empty() && narration.is_none() {
        return None;
    }

    let mut out = String::from("<activity_ledger>\n");
    out.push_str("Covered in the previous window (do NOT redo this work):\n");
    let skipped = entries.len().saturating_sub(MAX_ENTRIES);
    if skipped > 0 {
        out.push_str(&format!("- (+{skipped} earlier actions omitted)\n"));
    }
    for entry in &entries[skipped..] {
        out.push_str("- ");
        out.push_str(entry);
        out.push('\n');
    }
    if let Some(text) = narration {
        out.push_str("Where you left off: ");
        out.push_str(&text);
        out.push('\n');
    }
    out.push_str("Continue from here; re-read a file only for details this ledger lacks.\n");
    out.push_str("</activity_ledger>");

    // Pathological growth guard (a 1200-char ledger is ~400 tokens).
    if out.chars().count() > MAX_LEDGER_CHARS {
        let head: String = out.chars().take(MAX_LEDGER_CHARS - 3).collect();
        return Some(format!("{head}..."));
    }
    Some(out)
}

/// Extract the artifact index (M3c, action-executor-design §6.1) from an
/// outgoing window's messages: the summary+reference lines that survive the
/// rotation while the bodies go with the archive. Bodies stay fetchable via
/// `obs.fetch('<id>')` — the index is their persistent part.
///
/// The recognised line is the engine's `artifact_line` shape (codexec
/// `artifacts.rs`, pinned example in both test suites — keep them in step):
///
/// ```text
/// artifact art_0001 (stdout, 240112 bytes): <summary> — obs.fetch('art_0001') for the full body
/// ```
///
/// Every message variant that carries prose is scanned — observation panes
/// arrive as shape-A user messages and tool results alike. Returns `None`
/// when the window spilled nothing.
pub fn extract_artifact_index(messages: &[ChatMessage]) -> Option<String> {
    let mut entries: Vec<(String, String)> = Vec::new(); // (id, line), recency order
    for msg in messages {
        for text in message_texts(msg) {
            for line in text.lines() {
                let Some((id, line)) = artifact_ref_line(line) else {
                    continue;
                };
                // Re-mention moves the entry to the end (most recent wins,
                // and one id occupies one line — never a stack of echoes).
                entries.retain(|(seen, _)| *seen != id);
                entries.push((id, line));
            }
        }
    }
    if entries.is_empty() {
        return None;
    }

    let skipped = entries.len().saturating_sub(MAX_INDEX_ENTRIES);
    let mut out = String::from("<artifact_index>\n");
    out.push_str("Spilled artifacts still fetchable from the previous window (fetch with obs.fetch('<id>') for the full body; print only the part you need):\n");
    if skipped > 0 {
        out.push_str(&format!("- (+{skipped} earlier artifacts omitted)\n"));
    }
    for (_, line) in &entries[skipped..] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("</artifact_index>");

    // Pathological growth guard (same shape as the ledger's).
    if out.chars().count() > MAX_INDEX_CHARS {
        let head: String = out.chars().take(MAX_INDEX_CHARS - 3).collect();
        return Some(format!("{head}..."));
    }
    Some(out)
}

/// One `artifact_line`-shaped line → `(id, trimmed line)`. The `obs.fetch('`
/// tail is part of the signature: a prose mention of "artifact art_…" without
/// the fetch instruction is not a reference line.
fn artifact_ref_line(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    let id = line.strip_prefix("artifact ")?.split_whitespace().next()?;
    if !id.starts_with("art_") || !line.contains("— obs.fetch('art_") {
        return None;
    }
    Some((id.to_string(), line.to_string()))
}

/// The prose carrier of every message variant (`Custom` carries structured
/// data, not prose — skipped).
fn message_texts(msg: &ChatMessage) -> Vec<&str> {
    match msg {
        ChatMessage::System { content, .. } => vec![content],
        ChatMessage::User { content, .. } => vec![content],
        ChatMessage::Assistant { content, .. } => content.iter().map(String::as_str).collect(),
        ChatMessage::Tool { content, .. } => vec![content],
        ChatMessage::Custom { .. } => Vec::new(),
    }
}

/// Extract the plan state from an outgoing window's messages.
///
/// `update_plan` is a display-only protocol whose calls *replace* the whole
/// checklist, so the live plan is the last call's `plan` plus the objective
/// of the most recent call that carried one — subsequent calls may omit the
/// objective and lean on the tool's session memory, which a fresh window
/// does not have, so a previously-injected `<plan>` block (the hint slot
/// echoes it into the window) supplies the fallback. Rendered verbatim — no
/// step-text truncation, no entry caps: unlike the ledger, the plan is the
/// task's spine and is never evicted. Returns the whole `<plan>` block for
/// the hint slot and the `notes/plan_state.md` mirror.
pub fn extract_plan_state(messages: &[ChatMessage]) -> Option<String> {
    use agent_base::{PlanItem, PlanStepStatus, UpdatePlanArgs};

    let mut objective: Option<String> = None;
    let mut hinted_objective: Option<String> = None;
    let mut plan: Option<Vec<PlanItem>> = None;

    for msg in messages {
        for text in message_texts(msg) {
            if let Some(obj) = hinted_plan_objective(text) {
                hinted_objective = Some(obj);
            }
        }
        let ChatMessage::Assistant {
            tool_calls: Some(calls),
            ..
        } = msg
        else {
            continue;
        };
        for call in calls {
            if call.name != "update_plan" {
                continue;
            }
            // A malformed or schema-breaking call is skipped exactly as the
            // live tool would reject it — the last valid state stands.
            let Ok(args) = serde_json::from_str::<UpdatePlanArgs>(&call.arguments) else {
                continue;
            };
            if let Some(obj) = args.objective {
                objective = Some(obj);
            }
            if !args.plan.is_empty() {
                plan = Some(args.plan);
            }
        }
    }

    let plan = plan?;
    let objective = objective.or(hinted_objective);

    let mut out = String::from("<plan>\n");
    out.push_str("Standing plan (update it with update_plan as you progress):\n");
    if let Some(obj) = objective {
        out.push_str("Objective: ");
        out.push_str(&obj);
        out.push('\n');
    }
    for item in plan {
        let status = match item.status {
            PlanStepStatus::Pending => "pending",
            PlanStepStatus::InProgress => "in_progress",
            PlanStepStatus::Completed => "completed",
        };
        out.push_str("- [");
        out.push_str(status);
        out.push_str("] ");
        out.push_str(&item.step);
        out.push('\n');
    }
    out.push_str("</plan>");
    Some(out)
}

/// The `Objective: ` line of a previously-injected `<plan>` block — the
/// objective-inheritance source when this window's calls omitted it.
fn hinted_plan_objective(text: &str) -> Option<String> {
    let start = text.find("<plan>")?;
    let region = &text[start..];
    let end = region.find("</plan>").unwrap_or(region.len());
    region[..end]
        .lines()
        .find_map(|line| line.strip_prefix("Objective: ").map(str::to_string))
}

/// Map one tool call to a ledger line. `None` = not worth recording
/// (context-management churn) or nothing extractable.
fn ledger_entry(tool: &str, args_json: &str) -> Option<String> {
    let line = match tool {
        // Work tools with a known argument shape.
        "read_file" => {
            let args = parse(args_json)?;
            str_field(&args, "path").map(|p| format!("read {p}{}", line_range(&args)))?
        }
        "write_file" => {
            let args = parse(args_json)?;
            str_field(&args, "path").map(|p| format!("wrote {p}"))?
        }
        "edit_file" => {
            let args = parse(args_json)?;
            str_field(&args, "path").map(|p| format!("edited {p}"))?
        }
        "execute_command" => {
            let args = parse(args_json)?;
            str_field(&args, "command")
                .map(|c| format!("ran `{}`", truncate_chars(&one_line(&c), 60)))?
        }
        "search_content" => {
            let args = parse(args_json)?;
            let pattern = str_field(&args, "pattern")?;
            match str_field(&args, "path") {
                Some(p) => format!("searched \"{pattern}\" in {p}"),
                None => format!("searched \"{pattern}\""),
            }
        }
        "repo_map" => {
            let args = parse(args_json)?;
            match str_field(&args, "path") {
                Some(p) => format!("mapped structure of {p}"),
                None => "mapped repo structure".to_string(),
            }
        }
        "spawn_agent" => {
            let args = parse(args_json)?;
            str_field(&args, "task_name")
                .or_else(|| str_field(&args, "name"))
                .map(|n| format!("spawned sub-agent {n}"))?
        }
        "notes_write_file" | "notes_append_to_file" => {
            let args = parse(args_json)?;
            str_field(&args, "path").map(|p| format!("note {p}"))?
        }
        // Private context-management churn — not work.
        "history_list_windows"
        | "history_list_items"
        | "history_read_item"
        | "history_search_contents"
        | "notes_list_files"
        | "notes_read_file"
        | "notes_search_contents" => {
            return None;
        }
        // Unknown work tool: record the name plus its first string argument
        // (good enough to identify the call; keeps the ledger future-proof).
        _ => {
            let args = parse(args_json)?;
            let first = args
                .as_object()
                .and_then(|o| o.values().find_map(|v| v.as_str()));
            match first {
                Some(v) => format!("{tool} {}", truncate_chars(&one_line(v), 40)),
                None => tool.to_string(),
            }
        }
    };
    Some(truncate_chars(&line, MAX_ENTRY_CHARS))
}

fn parse(args_json: &str) -> Option<serde_json::Value> {
    serde_json::from_str(args_json).ok()
}

fn str_field(args: &serde_json::Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// `(L10-60)` suffix when the read was paginated.
fn line_range(args: &serde_json::Value) -> String {
    let offset = args.get("offset").and_then(|v| v.as_u64());
    let limit = args.get("limit").and_then(|v| v.as_u64());
    match (offset, limit) {
        (Some(o), Some(l)) => format!("(L{}-{})", o, o + l),
        (Some(o), None) => format!("(L{o}-)"),
        _ => String::new(),
    }
}

/// Collapse whitespace to single spaces (CJK-safe: char-based).
fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Char-based truncation with an ASCII ellipsis (CJK fonts render `…`
/// double-width — never use it).
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{head}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::ToolCallMessage;

    fn assistant(content: &str, calls: Vec<ToolCallMessage>) -> ChatMessage {
        ChatMessage::Assistant {
            content: if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            },
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: if calls.is_empty() { None } else { Some(calls) },
        }
    }

    fn call(name: &str, args: serde_json::Value) -> ToolCallMessage {
        ToolCallMessage {
            id: "t1".into(),
            name: name.into(),
            arguments: args.to_string(),
        }
    }

    #[test]
    fn extracts_work_actions_in_order() {
        let msgs = vec![assistant(
            "我先看结构。",
            vec![
                call("repo_map", serde_json::json!({})),
                call("read_file", serde_json::json!({"path": "README.md"})),
                call(
                    "read_file",
                    serde_json::json!({"path": "src/main.rs", "offset": 10, "limit": 50}),
                ),
                call(
                    "execute_command",
                    serde_json::json!({"command": "cargo test --lib"}),
                ),
                call("edit_file", serde_json::json!({"path": "src/lib.rs"})),
            ],
        )];
        let ledger = extract_ledger(&msgs).unwrap();
        assert!(ledger.contains("<activity_ledger>"));
        assert!(ledger.contains("- mapped repo structure"));
        assert!(ledger.contains("- read README.md"));
        assert!(ledger.contains("- read src/main.rs(L10-60)"));
        assert!(ledger.contains("- ran `cargo test --lib`"));
        assert!(ledger.contains("- edited src/lib.rs"));
        assert!(ledger.contains("Where you left off: 我先看结构。"));
    }

    #[test]
    fn excludes_context_management_churn() {
        let msgs = vec![assistant(
            "",
            vec![
                call("history_list_windows", serde_json::json!({})),
                call("notes_list_files", serde_json::json!({})),
                call(
                    "notes_read_file",
                    serde_json::json!({"path": "thread_hint.md"}),
                ),
            ],
        )];
        assert!(extract_ledger(&msgs).is_none());
    }

    #[test]
    fn notes_writes_are_recorded() {
        let msgs = vec![assistant(
            "",
            vec![call(
                "notes_write_file",
                serde_json::json!({"path": "thread_hint.md", "content": "..."}),
            )],
        )];
        let ledger = extract_ledger(&msgs).unwrap();
        assert!(ledger.contains("- note thread_hint.md"));
    }

    #[test]
    fn dedups_repeated_reads() {
        let msgs = vec![assistant(
            "",
            vec![
                call("read_file", serde_json::json!({"path": "README.md"})),
                call("read_file", serde_json::json!({"path": "README.md"})),
                call("read_file", serde_json::json!({"path": "src/lib.rs"})),
            ],
        )];
        let ledger = extract_ledger(&msgs).unwrap();
        assert_eq!(ledger.matches("- read README.md").count(), 1);
        assert!(ledger.contains("- read src/lib.rs"));
    }

    #[test]
    fn keeps_most_recent_entries_when_overflowing() {
        let mut calls = Vec::new();
        for i in 0..40 {
            calls.push(call(
                "read_file",
                serde_json::json!({"path": format!("file_{i:02}.rs")}),
            ));
        }
        let msgs = vec![assistant("", calls)];
        let ledger = extract_ledger(&msgs).unwrap();
        assert!(ledger.contains("earlier actions omitted"));
        // The tail (most recent) survives; the head does not.
        assert!(!ledger.contains("file_00.rs"));
        assert!(ledger.contains("file_39.rs"));
    }

    #[test]
    fn long_command_and_narration_are_truncated() {
        let long_cmd = "x".repeat(300);
        let long_text = "思".repeat(300);
        let msgs = vec![assistant(
            &long_text,
            vec![call(
                "execute_command",
                serde_json::json!({"command": long_cmd}),
            )],
        )];
        let ledger = extract_ledger(&msgs).unwrap();
        assert!(ledger.contains("..."));
        assert!(ledger.chars().count() < 1000);
    }

    #[test]
    fn unknown_tool_falls_back_to_name_and_first_arg() {
        let msgs = vec![assistant(
            "",
            vec![call(
                "future_tool",
                serde_json::json!({"target": "something", "n": 3}),
            )],
        )];
        let ledger = extract_ledger(&msgs).unwrap();
        assert!(ledger.contains("- future_tool something"));
    }

    #[test]
    fn spawn_agent_uses_task_name() {
        let msgs = vec![assistant(
            "",
            vec![call(
                "spawn_agent",
                serde_json::json!({"task_name": "analyzer", "task": "read the code"}),
            )],
        )];
        let ledger = extract_ledger(&msgs).unwrap();
        assert!(ledger.contains("- spawned sub-agent analyzer"));
    }

    #[test]
    fn empty_window_yields_none() {
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            assistant("", vec![]),
        ];
        assert!(extract_ledger(&msgs).is_none());
    }

    // ---- the artifact index (M3c) ----

    /// The pinned model-face line — the same example codexec's `artifact_line`
    /// test pins. If that shape changes, change this matcher in the same
    /// breath.
    #[test]
    fn artifact_index_carries_the_pinned_reference_line() {
        let line = "artifact art_0001 (stdout, 11 bytes): first body — obs.fetch('art_0001') for the full body";
        let msgs = vec![ChatMessage::user(format!("[observation]\n{line}"))];
        let index = extract_artifact_index(&msgs).unwrap();
        assert!(index.contains("<artifact_index>"));
        assert!(index.contains(line), "{index}");
        assert!(index.contains("obs.fetch('<id>')"), "{index}");
    }

    #[test]
    fn artifact_index_scans_tool_results_and_dedups_by_id() {
        let line = "artifact art_0002 (observation, 40 bytes): body — obs.fetch('art_0002') for the full body";
        let msgs = vec![
            ChatMessage::user(format!("shape-A pane\n{line}")),
            ChatMessage::Tool {
                tool_call_id: "t1".into(),
                name: Some("code_cell".into()),
                content: format!("tool result\n{line}"),
                images: Vec::new(),
            },
        ];
        let index = extract_artifact_index(&msgs).unwrap();
        assert_eq!(
            index.matches("artifact art_0002").count(),
            1,
            "one id, one line — never a stack of echoes: {index}"
        );
    }

    #[test]
    fn artifact_index_keeps_the_most_recent_entries() {
        let mut msgs = Vec::new();
        for i in 1..=12 {
            let line = format!(
                "artifact art_{i:04} (stdout, 1 bytes): s — obs.fetch('art_{i:04}') for the full body"
            );
            msgs.push(ChatMessage::user(line));
        }
        let index = extract_artifact_index(&msgs).unwrap();
        assert!(index.contains("earlier artifacts omitted"), "{index}");
        assert!(!index.contains("artifact art_0001 ("), "{index}");
        assert!(index.contains("artifact art_0012 ("), "{index}");
    }

    #[test]
    fn a_prose_mention_without_the_fetch_instruction_is_not_a_reference() {
        let msgs = vec![
            ChatMessage::user("we minted artifact art_0001 yesterday"),
            assistant("artifact art_0002 is what we call them", vec![]),
        ];
        assert!(extract_artifact_index(&msgs).is_none());
    }

    /// The summarized real shape (acceptance 20261029_a8015baa): a big body's
    /// line carries the marker in the middle and is ONE line — that is what
    /// the extractor matches. (When codexec's `summary_of` leaked newlines
    /// into this line, real artifacts silently dropped out of the index; the
    /// one-line invariant is pinned there too.)
    #[test]
    fn a_summarized_real_shape_line_is_an_index_entry() {
        let line = "artifact art_0002 (stdout, 200001 bytes): BBBB BBBB ...[output truncated] BBBB BBBB — obs.fetch('art_0002') for the full body";
        let msgs = vec![ChatMessage::user(line.to_string())];
        let index = extract_artifact_index(&msgs).unwrap();
        assert!(index.contains(line), "{index}");
    }

    // ── extract_plan_state: the plan survives rotation, verbatim ──

    #[test]
    fn plan_state_reads_the_last_call_and_inherits_the_objective() {
        let msgs = vec![assistant(
            "",
            vec![
                call(
                    "update_plan",
                    serde_json::json!({
                        "objective": "Install the Casdoor authentication system",
                        "plan": [
                            {"step": "Scaffold the module", "status": "completed"},
                            {"step": "Wire the provider", "status": "pending"},
                        ],
                    }),
                ),
                // Follow-up calls may omit the objective (the tool remembers
                // it) and replace the whole checklist.
                call(
                    "update_plan",
                    serde_json::json!({
                        "plan": [{"step": "Wire the provider", "status": "in_progress"}],
                    }),
                ),
            ],
        )];
        let plan = extract_plan_state(&msgs).unwrap();
        assert!(
            plan.contains("Objective: Install the Casdoor authentication system"),
            "{plan}"
        );
        assert!(plan.contains("- [in_progress] Wire the provider"), "{plan}");
        assert!(
            !plan.contains("Scaffold the module"),
            "old steps are replaced: {plan}"
        );
    }

    #[test]
    fn plan_state_objective_falls_back_to_the_previous_hint_block() {
        let prior_hint = "<plan>\nStanding plan (update it with update_plan as you progress):\n\
             Objective: Ship the v2 landing page\n- [pending] a\n</plan>";
        let msgs = vec![
            ChatMessage::system(prior_hint),
            assistant(
                "",
                vec![call(
                    "update_plan",
                    serde_json::json!({
                        "plan": [{"step": "b", "status": "in_progress"}],
                    }),
                )],
            ),
        ];
        let plan = extract_plan_state(&msgs).unwrap();
        assert!(
            plan.contains("Objective: Ship the v2 landing page"),
            "{plan}"
        );
        assert!(plan.contains("- [in_progress] b"), "{plan}");
    }

    #[test]
    fn plan_state_skips_malformed_calls_like_the_live_tool() {
        let msgs = vec![assistant(
            "",
            vec![
                call(
                    "update_plan",
                    serde_json::json!({"plan": [{"step": "keep me", "status": "pending"}]}),
                ),
                ToolCallMessage {
                    id: "tc_bad".into(),
                    name: "update_plan".into(),
                    // Truncated arguments (the mimo truncation pathology).
                    arguments: "{\"plan\": [{\"step\": \"dro".into(),
                },
            ],
        )];
        let plan = extract_plan_state(&msgs).unwrap();
        assert!(plan.contains("- [pending] keep me"), "{plan}");
    }

    #[test]
    fn plan_state_is_none_without_a_valid_plan_call() {
        let msgs = vec![
            assistant(
                "",
                vec![call("read_file", serde_json::json!({"path": "a"}))],
            ),
            assistant("", vec![]),
        ];
        assert!(extract_plan_state(&msgs).is_none());
    }

    #[test]
    fn plan_state_step_text_is_verbatim() {
        let long = "x".repeat(300);
        let msgs = vec![assistant(
            "",
            vec![call(
                "update_plan",
                serde_json::json!({"plan": [{"step": long, "status": "pending"}]}),
            )],
        )];
        let plan = extract_plan_state(&msgs).unwrap();
        assert!(plan.contains(&long), "step text must not be truncated");
    }
}
