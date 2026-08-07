use std::sync::Arc;

use agent_base::{AgentResult, Tool, ToolContext, ToolControlFlow, ToolOutput};
use agent_works::skill::SkillRegistry;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Tool that lets the LLM list all available skills with their status.
///
/// Returns skill names, descriptions, and metadata. Supports optional
/// category filtering.
pub struct ListSkillsTool {
    registry: Arc<SkillRegistry>,
}

impl ListSkillsTool {
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for ListSkillsTool {
    fn name(&self) -> &'static str {
        "list_skills"
    }

    fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "list_skills",
                "description": "列出所有可用技能及其状态。可按分类筛选。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "可选，按分类筛选技能 (e.g. 'ops', 'testing')"
                        }
                    }
                }
            }
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<ToolOutput> {
        let category_filter = args
            .get("category")
            .and_then(Value::as_str)
            .map(|s| s.to_string());

        let all_skills = self.registry.list().await;

        let filtered: Vec<_> = if let Some(ref cat) = category_filter {
            all_skills.into_iter().filter(|s| s.category == *cat).collect()
        } else {
            all_skills
        };

        if filtered.is_empty() {
            let msg = if category_filter.is_some() {
                format!("没有找到匹配分类的技能")
            } else {
                "当前没有注册任何技能".to_string()
            };
            return Ok(ToolOutput {
                summary: msg,
                raw: Some(json!({"skills": []})),
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
        }

        let skill_list: Vec<Value> = filtered
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "description": s.description,
                    "category": s.category,
                    "version": s.version,
                    "tags": s.tags,
                    "has_plan": s.has_plan,
                    "param_count": s.param_defs.len(),
                })
            })
            .collect();

        let summary = format!(
            "可用技能 ({} 个):\n{}",
            skill_list.len(),
            filtered
                .iter()
                .map(|s| {
                    let plan_mark = if s.has_plan { " [模板]" } else { "" };
                    format!("- **{}**{}: {}", s.name, plan_mark, s.description)
                })
                .collect::<Vec<_>>()
                .join("\n")
        );

        tracing::debug!(
            count = skill_list.len(),
            category = ?category_filter,
            "list_skills called"
        );

        Ok(ToolOutput {
            summary,
            raw: Some(json!({"skills": skill_list})),
            control_flow: ToolControlFlow::Break,
            truncation: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_works::skill::Skill;
    use std::sync::Arc;

    struct TestSkill {
        name: &'static str,
        desc: &'static str,
    }

    impl Skill for TestSkill {
        fn name(&self) -> &'static str {
            self.name
        }
        fn brief_description(&self) -> String {
            self.desc.to_string()
        }
        fn detailed_description(&self) -> String {
            format!("Detailed: {}", self.desc)
        }
        fn tools(&self) -> Vec<Arc<dyn agent_base::Tool>> {
            vec![]
        }
    }

    #[tokio::test]
    async fn test_list_skills_empty() {
        let registry = Arc::new(SkillRegistry::new());
        let tool = ListSkillsTool::new(registry);
        let result = tool.call(&json!({}), &dummy_ctx()).await.unwrap();
        assert!(result.summary.contains("没有注册任何技能"));
    }

    #[tokio::test]
    async fn test_list_skills_with_skills() {
        let registry = Arc::new(SkillRegistry::new());
        registry
            .register(Arc::new(TestSkill {
                name: "deploy",
                desc: "Deploy the app",
            }))
            .await;
        registry
            .register(Arc::new(TestSkill {
                name: "review",
                desc: "Review code",
            }))
            .await;

        let tool = ListSkillsTool::new(registry);
        let result = tool.call(&json!({}), &dummy_ctx()).await.unwrap();

        assert!(result.summary.contains("deploy"));
        assert!(result.summary.contains("review"));

        let raw = result.raw.unwrap();
        let skills = raw["skills"].as_array().unwrap();
        assert_eq!(skills.len(), 2);
    }

    #[tokio::test]
    async fn test_list_skills_category_filter() {
        let registry = Arc::new(SkillRegistry::new());
        registry
            .register(Arc::new(TestSkill {
                name: "deploy",
                desc: "Deploy",
            }))
            .await;

        // No category match
        let tool = ListSkillsTool::new(registry);
        // Note: category filter won't match because TestSkill uses default empty category
        let result = tool
            .call(&json!({"category": "nonexistent"}), &dummy_ctx())
            .await
            .unwrap();
        assert!(result.summary.contains("没有找到"));
    }

    fn dummy_ctx() -> ToolContext {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ToolContext {
            session_id: agent_base::SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: agent_base::Language::Zh,
            cancel_token: tokio_util::sync::CancellationToken::new(),
        }
    }
}
