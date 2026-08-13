//! Multi-agent kernel tools.
//!
//! Six tools that the LLM uses to manage sub-agents:
//! spawn, send_message, followup_task, wait, list, close.
//!
//! Each tool holds an `Arc<MultiAgentRuntime>` and delegates to its methods.

use std::sync::Arc;

use agent_base::Tool;
use agent_works::multi_agent::MultiAgentRuntime;

mod close_agent;
mod followup_task;
mod list_agents;
mod send_message;
mod spawn_agent;
mod wait_agent;

pub use close_agent::{CloseAgentArgs, CloseAgentOutput, CloseAgentTool};
pub use followup_task::{FollowupTaskArgs, FollowupTaskOutput, FollowupTaskTool};
pub use list_agents::{ListAgentItem, ListAgentsArgs, ListAgentsTool};
pub use send_message::{SendMessageArgs, SendMessageOutput, SendMessageTool};
pub use spawn_agent::{SpawnAgentArgs, SpawnAgentOutput, SpawnAgentTool};
pub use wait_agent::{WaitAgentArgs, WaitAgentOutput, WaitAgentTool};

/// Create all 6 multi-agent tools, sharing the same runtime.
pub fn create_all_tools(runtime: Arc<MultiAgentRuntime>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(SpawnAgentTool::new(runtime.clone())),
        Arc::new(SendMessageTool::new(runtime.clone())),
        Arc::new(FollowupTaskTool::new(runtime.clone())),
        Arc::new(WaitAgentTool::new(runtime.clone())),
        Arc::new(ListAgentsTool::new(runtime.clone())),
        Arc::new(CloseAgentTool::new(runtime)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::{Language, LlmClient, TypedTool};
    use agent_works::multi_agent::MultiAgentConfig;
    use std::pin::Pin;
    use tokio_util::sync::CancellationToken;

    // ── Minimal mock LLM client ──

    struct StubClient;

    #[async_trait::async_trait]
    impl LlmClient for StubClient {
        async fn chat(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> agent_base::AgentResult<serde_json::Value> {
            Ok(serde_json::json!({"choices": [{"message": {"content": "ok"}}]}))
        }

        async fn chat_stream(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> agent_base::AgentResult<
            Pin<
                Box<
                    dyn futures_core::Stream<
                            Item = agent_base::AgentResult<agent_base::StreamChunk>,
                        > + Send,
                >,
            >,
        > {
            let chunks: Vec<agent_base::AgentResult<agent_base::StreamChunk>> = vec![
                Ok(agent_base::StreamChunk::Text("ok".to_string())),
                Ok(agent_base::StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ];
            Ok(Box::pin(futures_util::stream::iter(chunks)))
        }

        fn capabilities(&self) -> agent_base::LlmCapabilities {
            agent_base::LlmCapabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: false,
                supports_thinking: false,
                max_context_tokens: None,
                max_output_tokens: None,
            }
        }
    }

    fn make_runtime() -> Arc<MultiAgentRuntime> {
        let client = agent_base::llm::adapt(Arc::new(StubClient));
        let cancel = CancellationToken::new();
        Arc::new(MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            client,
            vec![],
            cancel,
            None,
            Language::En,
        ))
    }

    fn make_tool_ctx() -> agent_base::ToolContext {
        agent_base::ToolContext::for_test()
    }

    // ── name / description / schema ──

    #[test]
    fn test_spawn_agent_tool_metadata() {
        let t = SpawnAgentTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "spawn_agent");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.schema();
        assert_eq!(schema["type"], "object");
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"task_name".into())
        );
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"message".into())
        );
    }

    #[test]
    fn test_send_message_tool_metadata() {
        let t = SendMessageTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "send_message");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.schema();
        assert_eq!(schema["type"], "object");
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"agent_path".into())
        );
    }

    #[test]
    fn test_followup_task_tool_metadata() {
        let t = FollowupTaskTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "followup_task");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.schema();
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"agent_path".into())
        );
    }

    #[test]
    fn test_wait_agent_tool_metadata() {
        let t = WaitAgentTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "wait_agent");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.schema();
        assert_eq!(schema["type"], "object");
        // All fields are optional, so schemars omits the `required` key.
        assert!(schema.get("required").is_none());
    }

    #[test]
    fn test_list_agents_tool_metadata() {
        let t = ListAgentsTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "list_agents");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.schema();
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn test_close_agent_tool_metadata() {
        let t = CloseAgentTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "close_agent");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.schema();
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"agent_path".into())
        );
    }

    // ── format_output ──

    #[test]
    fn test_spawn_agent_format_output() {
        let t = SpawnAgentTool::new(make_runtime());
        let out = t.format_output(SpawnAgentOutput {
            agent_path: "root/w1".into(),
            message: "ok".into(),
        });
        let text = agent_base::tool::content_text(&[out]);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["agent_path"], "root/w1");
        assert_eq!(v["message"], "ok");
    }

    #[test]
    fn test_send_message_format_output() {
        let t = SendMessageTool::new(make_runtime());
        let out = t.format_output(SendMessageOutput { delivered: true });
        let text = agent_base::tool::content_text(&[out]);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["delivered"], true);
    }

    #[test]
    fn test_followup_task_format_output() {
        let t = FollowupTaskTool::new(make_runtime());
        let out = t.format_output(FollowupTaskOutput {
            accepted: true,
            agent_path: "root/w1".into(),
        });
        let text = agent_base::tool::content_text(&[out]);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["accepted"], true);
    }

    #[test]
    fn test_wait_agent_format_output() {
        let t = WaitAgentTool::new(make_runtime());
        let out = t.format_output(WaitAgentOutput {
            status: "timeout".into(),
            result: None,
            agent_path: None,
            has_more: false,
        });
        let text = agent_base::tool::content_text(&[out]);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["status"], "timeout");
        assert_eq!(v["has_more"], false);
    }

    #[test]
    fn test_close_agent_format_output() {
        let t = CloseAgentTool::new(make_runtime());
        let out = t.format_output(CloseAgentOutput {
            closed: true,
            previous_status: "idle".into(),
            message: "done".into(),
        });
        let text = agent_base::tool::content_text(&[out]);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["closed"], true);
    }

    // ── call_typed for tools that don't need spawn ──

    #[tokio::test]
    async fn test_list_agents_call_empty() {
        let rt = make_runtime();
        let t = ListAgentsTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t.call_typed(ListAgentsArgs {}, &ctx).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_close_agent_nonexistent() {
        let rt = make_runtime();
        let t = CloseAgentTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(
                CloseAgentArgs {
                    agent_path: "root/ghost".into(),
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.closed);
        assert_eq!(result.previous_status, "unknown");
    }

    #[tokio::test]
    async fn test_send_message_nonexistent() {
        let rt = make_runtime();
        let t = SendMessageTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(
                SendMessageArgs {
                    agent_path: "root/ghost".into(),
                    message: "hi".into(),
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.delivered);
    }

    #[tokio::test]
    async fn test_followup_task_nonexistent() {
        let rt = make_runtime();
        let t = FollowupTaskTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(
                FollowupTaskArgs {
                    agent_path: "root/ghost".into(),
                    task: "do".into(),
                    interrupt: true,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.accepted);
    }

    // ── send_message + followup_task + wait result round-trip ──

    #[tokio::test]
    async fn test_send_message_and_followup_task_roundtrip() {
        let rt = make_runtime();

        // Spawn a child first
        let path = rt
            .spawn_child("worker", "you are a worker".into(), 1, 0, vec![])
            .await
            .unwrap();

        // Send a message (no execution trigger)
        let t = SendMessageTool::new(rt.clone());
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(
                SendMessageArgs {
                    agent_path: path.clone(),
                    message: "context info".into(),
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(result.delivered);

        // Send a task (triggers execution, drains pending messages)
        let t2 = FollowupTaskTool::new(rt.clone());
        let result2 = t2
            .call_typed(
                FollowupTaskArgs {
                    agent_path: path.clone(),
                    task: "do work".into(),
                    interrupt: true,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(result2.accepted);

        // Wait for result
        let t3 = WaitAgentTool::new(rt.clone());
        let result3 = t3
            .call_typed(
                WaitAgentArgs {
                    agent_path: Some(path.clone()),
                    timeout_ms: 5000,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(result3.status, "ok");
        assert!(result3.result.is_some());

        // Close
        let t4 = CloseAgentTool::new(rt.clone());
        let result4 = t4
            .call_typed(
                CloseAgentArgs {
                    agent_path: path.clone(),
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(result4.closed);
    }

    // ── spawn_agent call_typed ──

    #[tokio::test]
    async fn test_spawn_agent_call() {
        let rt = make_runtime();
        let t = SpawnAgentTool::new(rt.clone());
        let ctx = make_tool_ctx();

        let result = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "helper".into(),
                    message: "do something".into(),
                    agent_type: None,
                    system_prompt: Some("you are a helper".into()),
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                    depth: 1,
                },
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(result.agent_path, "root/helper");
        assert!(result.message.contains("spawned"));

        // Verify the agent shows up in list
        let t2 = ListAgentsTool::new(rt);
        let list = t2.call_typed(ListAgentsArgs {}, &ctx).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].agent_path, "root/helper");
    }

    // ── spawn with auto-generated system prompt from agent_type ──

    #[tokio::test]
    async fn test_spawn_agent_with_agent_type() {
        let rt = make_runtime();
        let t = SpawnAgentTool::new(rt);
        let ctx = make_tool_ctx();

        let result = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "searcher".into(),
                    message: "search for info".into(),
                    agent_type: Some("researcher".into()),
                    system_prompt: None,
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                    depth: 1,
                },
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(result.agent_path, "root/searcher");
    }

    // ── spawn limit exceeded ──

    #[tokio::test]
    async fn test_spawn_agent_limit_exceeded() {
        let client = agent_base::llm::adapt(Arc::new(StubClient));
        let cancel = CancellationToken::new();
        let config = MultiAgentConfig {
            enabled: true,
            max_sub_agents: 1,
            max_agent_depth: 1,
        };
        let rt = Arc::new(MultiAgentRuntime::new(
            config,
            client,
            vec![],
            cancel,
            None,
            Language::En,
        ));

        let t = SpawnAgentTool::new(rt.clone());
        let ctx = make_tool_ctx();

        // First spawn succeeds
        let r1 = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "first".into(),
                    message: "task".into(),
                    agent_type: None,
                    system_prompt: None,
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                    depth: 1,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(r1.agent_path, "root/first");

        // Second spawn fails
        let r2 = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "second".into(),
                    message: "task".into(),
                    agent_type: None,
                    system_prompt: None,
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                    depth: 1,
                },
                &ctx,
            )
            .await
            .unwrap();

        assert!(r2.agent_path.is_empty());
        assert!(r2.message.contains("Failed"));
    }

    // ── create_all_tools ──

    #[test]
    fn test_create_all_tools_returns_six() {
        let rt = make_runtime();
        let tools = create_all_tools(rt);
        assert_eq!(tools.len(), 6);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"spawn_agent"));
        assert!(names.contains(&"send_message"));
        assert!(names.contains(&"followup_task"));
        assert!(names.contains(&"wait_agent"));
        assert!(names.contains(&"list_agents"));
        assert!(names.contains(&"close_agent"));
    }
}
