//! Skill kernel tools.
//!
//! Tools that the LLM uses to interact with the skill system:
//! apply_skill, get_skill_detail.

mod apply_tool;
mod detail_tool;

pub use apply_tool::ApplySkillTool;
pub use detail_tool::SkillDetailTool;
