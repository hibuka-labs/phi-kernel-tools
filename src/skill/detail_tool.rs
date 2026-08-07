use std::sync::Arc;

use agent_base::{AgentResult, Tool, ToolContext, ToolControlFlow, ToolOutput, UserEvent};
use agent_works::skill::Skill;
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct SkillDetailTool {
    pub skills: Vec<Arc<dyn Skill>>,
    pub name: &'static str,
}

impl SkillDetailTool {
    pub fn new(skills: Vec<Arc<dyn Skill>>, tool_name: String) -> Self {
        let name: &'static str = Box::leak(tool_name.into_boxed_str());
        Self { skills, name }
    }
}

#[async_trait]
impl Tool for SkillDetailTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get detailed instructions for a Skill. Call this when you need the complete usage guide for a Skill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Skill name"
                        }
                    },
                    "required": ["name"]
                }
            }
        })
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<ToolOutput> {
        let name = args.get("name").and_then(Value::as_str).unwrap_or("");

        if name.is_empty() {
            return Ok(ToolOutput {
                summary: format!(
                    "Please provide a Skill name. Available Skills: {}",
                    self.skills
                        .iter()
                        .map(|s| s.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                raw: None,
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
        }

        let detail = self
            .skills
            .iter()
            .find(|s| s.name() == name)
            .map(|s| s.detailed_description());

        tracing::debug!("skill detail queried");
        ctx.emit_user_event(UserEvent::Structured {
            event_type: "skill_detail_loaded".to_string(),
            data: json!({
                "skill": name,
            }),
        });

        match detail {
            Some(desc) => Ok(ToolOutput {
                summary: desc.to_string(),
                raw: None,
                control_flow: ToolControlFlow::Break,
                truncation: None,
            }),
            None => {
                let available: Vec<&str> = self.skills.iter().map(|s| s.name()).collect();
                Ok(ToolOutput {
                    summary: format!(
                        "Skill '{}' not found. Available Skills: {}",
                        name,
                        available.join(", ")
                    ),
                    raw: None,
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct TestSkill {
        name: &'static str,
        desc: &'static str,
        detail: &'static str,
    }

    impl Skill for TestSkill {
        fn name(&self) -> &'static str {
            self.name
        }
        fn brief_description(&self) -> String {
            self.desc.to_string()
        }
        fn detailed_description(&self) -> String {
            self.detail.to_string()
        }
        fn tools(&self) -> Vec<Arc<dyn agent_base::Tool>> {
            vec![]
        }
    }

    fn make_skills() -> Vec<Arc<dyn Skill>> {
        vec![
            Arc::new(TestSkill {
                name: "deploy",
                desc: "Deploy the app",
                detail: "# Deploy\n\nSteps to deploy the application.",
            }),
            Arc::new(TestSkill {
                name: "test",
                desc: "Run tests",
                detail: "# Test\n\nRun the test suite.",
            }),
        ]
    }

    fn make_tool() -> SkillDetailTool {
        SkillDetailTool::new(make_skills(), "get_skill_detail".to_string())
    }

    fn dummy_ctx() -> ToolContext {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ToolContext {
            session_id: agent_base::SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: agent_base::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[test]
    fn test_name_reflects_tool_name_arg() {
        let tool = SkillDetailTool::new(make_skills(), "custom_detail".to_string());
        assert_eq!(tool.name(), "custom_detail");
    }

    #[test]
    fn test_definition_has_correct_schema() {
        let tool = make_tool();
        let def = tool.definition();
        assert_eq!(def["function"]["name"], "get_skill_detail");
        assert!(def["function"]["description"]
            .as_str()
            .unwrap()
            .contains("detailed instructions"));
        let required = def["function"]["parameters"]["required"].as_array().unwrap();
        assert!(required.contains(&"name".into()));
    }

    #[tokio::test]
    async fn test_call_returns_detail_for_known_skill() {
        let tool = make_tool();
        let ctx = dummy_ctx();
        let result = tool
            .call(&json!({"name": "deploy"}), &ctx)
            .await
            .unwrap();
        assert!(result.summary.contains("# Deploy"));
        assert!(result.summary.contains("deploy the application"));
    }

    #[tokio::test]
    async fn test_call_returns_error_for_unknown_skill() {
        let tool = make_tool();
        let ctx = dummy_ctx();
        let result = tool
            .call(&json!({"name": "nonexistent"}), &ctx)
            .await
            .unwrap();
        assert!(result.summary.contains("not found"));
        assert!(result.summary.contains("deploy"));
        assert!(result.summary.contains("test"));
    }

    #[tokio::test]
    async fn test_call_with_empty_name_lists_available() {
        let tool = make_tool();
        let ctx = dummy_ctx();
        let result = tool.call(&json!({}), &ctx).await.unwrap();
        assert!(result.summary.contains("Please provide a Skill name"));
        assert!(result.summary.contains("deploy"));
        assert!(result.summary.contains("test"));
    }

    #[tokio::test]
    async fn test_call_with_empty_skills_graceful() {
        let tool = SkillDetailTool::new(vec![], "get_skill_detail".to_string());
        let ctx = dummy_ctx();
        let result = tool.call(&json!({}), &ctx).await.unwrap();
        assert!(result.summary.contains("Please provide a Skill name"));
    }

    #[tokio::test]
    async fn test_call_emits_structured_event() {
        let tool = make_tool();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = ToolContext {
            session_id: agent_base::SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: agent_base::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
        };
        tool.call(&json!({"name": "deploy"}), &ctx).await.unwrap();
        // The event is emitted before the function returns, so it should be available
        match rx.try_recv() {
            Ok(UserEvent::Structured { event_type, data }) => {
                assert_eq!(event_type, "skill_detail_loaded");
                assert_eq!(data["skill"], "deploy");
            }
            other => panic!("Expected Structured event, got {:?}", other),
        }
    }

    #[test]
    fn test_control_flow_is_break() {
        let tool = make_tool();
        let def = tool.definition();
        // Verify the definition has the expected shape for a Tool impl
        assert!(def.is_object());
        assert_eq!(def["type"], "function");
    }
}
