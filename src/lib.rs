//! Kernel tools for the phi-agent framework.
//!
//! This crate provides the Tool implementations that the LLM uses to interact
//! with agent-works infrastructure:
//!
//! - **multi-agent**: spawn, send_message, followup_task, wait, list, close
//! - **skill**: apply_skill, get_skill_detail
//!
//! Each tool implements `agent_base::Tool` (or `TypedTool`) and delegates to
//! the corresponding `agent_works` infrastructure types.
//!
//! # Feature gates
//!
//! | Feature | Tools provided |
//! |---------|---------------|
//! | `multi-agent` | 6 multi-agent tools |
//! | `skill` | ApplySkillTool, SkillDetailTool |
//!
//! All features are enabled by default. Use `default-features = false` to
//! selectively exclude kernel tools you don't need.

#[cfg(feature = "multi-agent")]
pub mod multi_agent;

#[cfg(feature = "skill")]
pub mod skill;
