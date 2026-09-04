//! Multi-agent kernel tools.
//!
//! Four tools that the LLM uses to manage sub-agents: spawn, send_message,
//! list, close. There is deliberately **no** `wait_agent` tool: child results
//! are pushed into the parent's context automatically (watcher → TUI), and
//! the parent "waits" by simply ending its turn. A blocking-wait tool would
//! hand the model a truncation-prone escape hatch that bypasses the push
//! pipeline. (`followup_task` still exists as a deprecated shim —
//! it forwards to `send_message(trigger=true)` and is kept for one version
//! outside this factory, design doc §8.3.)
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

pub use close_agent::{CloseAgentArgs, CloseAgentOutput, CloseAgentTool};
#[allow(deprecated)]
pub use followup_task::{FollowupTaskArgs, FollowupTaskOutput, FollowupTaskTool};
pub use list_agents::{ListAgentItem, ListAgentsArgs, ListAgentsTool};
pub use send_message::{SendMessageArgs, SendMessageOutput, SendMessageTool};
pub use spawn_agent::{SpawnAgentArgs, SpawnAgentOutput, SpawnAgentTool};

/// Create the 4 active multi-agent tools, sharing the same runtime.
///
/// `workspace_root` is injected into each child's system prompt as a plain
/// fact ("Working directory: …") — children share the parent's process cwd
/// and otherwise cannot discover how their relative paths resolve.
///
/// No `wait_agent`: results are pushed to the parent automatically; the
/// parent waits by ending its turn. `followup_task` is also **not** here
/// (§8.3): its trigger semantics now live in `send_message(trigger=true)`.
pub fn create_all_tools(
    runtime: Arc<MultiAgentRuntime>,
    workspace_root: std::path::PathBuf,
) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(SpawnAgentTool::new(runtime.clone(), workspace_root)),
        Arc::new(SendMessageTool::new(runtime.clone())),
        Arc::new(ListAgentsTool::new(runtime.clone())),
        Arc::new(CloseAgentTool::new(runtime)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::llm_trait::{
        Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
    };
    use agent_base::{Language, TypedTool};
    use agent_works::multi_agent::MultiAgentConfig;
    use tokio_util::sync::CancellationToken;

    // ── Minimal mock LLM client ──

    struct StubClient;

    #[async_trait::async_trait]
    impl LlmProvider for StubClient {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            let chunks = vec![
                Ok(agent_base::StreamChunk::Text("ok".to_string())),
                Ok(agent_base::StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ];
            Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
                chunks,
            ))))
        }

        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: "ok".to_string(),
                reasoning_content: None,
                thinking_signature: None,
                tool_calls: vec![],
                finish_reason: agent_base::llm::FinishReason::Stop,
                usage: agent_base::UsageInfo::default(),
                raw: None,
            })
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: false,
                supports_thinking: false,
                max_context_tokens: None,
                max_output_tokens: None,
            }
        }

        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                name: "stub".to_string(),
                model: "test".to_string(),
                version: None,
            }
        }
    }

    fn make_runtime() -> Arc<MultiAgentRuntime> {
        let client: Arc<dyn agent_base::llm_trait::LlmProvider> = Arc::new(StubClient);
        let cancel = CancellationToken::new();
        Arc::new(MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            client,
            vec![],
            cancel,
            None,
            Language::En,
            None,
            None,
        ))
    }

    fn make_tool_ctx() -> agent_base::ToolContext {
        agent_base::ToolContext::for_test()
    }

    // ── name / description / schema ──

    #[test]
    fn test_spawn_agent_tool_metadata() {
        let t = SpawnAgentTool::new(make_runtime(), std::env::current_dir().unwrap());
        assert_eq!(agent_base::TypedTool::name(&t), "spawn_agent");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.schema();
        assert_eq!(schema["type"], "object");
        // Minimal LLM-facing schema: task_name + task are required;
        // fork_turns / model are optional overrides. depth/full_permission
        // stay config-driven and never appear in the schema.
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&"task_name".into()));
        assert!(required.contains(&"task".into()));
        let props = schema["properties"].as_object().unwrap();
        assert_eq!(props.len(), 4, "schema must stay minimal: {props:?}");
        assert!(props.contains_key("task"));
        assert!(props.contains_key("task_name"));
        assert!(props.contains_key("fork_turns"));
        assert!(props.contains_key("model"));
        // Removed fields must stay out of the schema.
        assert!(!props.contains_key("system_prompt"));
        assert!(!props.contains_key("message"));
        assert!(!props.contains_key("agent_type"));
        assert!(!props.contains_key("fork_history"));
        assert!(!props.contains_key("depth"));
        assert!(!props.contains_key("full_permission"));
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
    #[allow(deprecated)]
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
        // The dead `interrupt` flag is off the schema (K2 / §8.3).
        assert!(
            !schema["properties"]
                .as_object()
                .unwrap()
                .contains_key("interrupt")
        );
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
        let t = SpawnAgentTool::new(make_runtime(), std::env::current_dir().unwrap());
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
    #[allow(deprecated)]
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
    fn test_close_agent_format_output() {
        let t = CloseAgentTool::new(make_runtime());
        let out = t.format_output(CloseAgentOutput {
            closed: true,
            previous_status: "done".into(),
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
                    trigger: false,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.delivered);
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_followup_task_nonexistent() {
        let rt = make_runtime();
        let t = FollowupTaskTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(
                FollowupTaskArgs {
                    agent_path: "root/ghost".into(),
                    task: "do".into(),
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.accepted);
    }

    // ── send_message + followup_task round-trip ──

    #[tokio::test]
    async fn test_send_message_and_followup_task_roundtrip() {
        let rt = make_runtime();

        // Spawn a child first
        let path = rt
            .spawn_child("worker", "you are a worker".into() , 1, false, vec![])
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
                    trigger: false,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(result.delivered);

        // Send a task via the NEW trigger flag (send_message(trigger=true)
        // replaces followup_task; the deprecated shim forwards here).
        let result2 = t
            .call_typed(
                SendMessageArgs {
                    agent_path: path.clone(),
                    message: "do work".into(),
                    trigger: true,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert!(result2.delivered);

        // Result arrives via the runtime's push pipeline; the LLM-facing
        // wait_agent tool is gone — collect it through the internal API.
        let result3 = rt.wait_for_result(Some(&path), 5000).await;
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
    //
    // The behavioural matrix (aliases, three-level fallback, B5, presets)
    // lives in spawn_agent.rs's own tests; these cover factory-level wiring.

    #[tokio::test]
    async fn test_spawn_agent_call() {
        let rt = make_runtime();
        let t = SpawnAgentTool::new(rt.clone(), std::env::current_dir().unwrap());
        let ctx = make_tool_ctx();

        let result = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "helper".into(),
                    task: "do something useful".into(),
                    fork_turns: None,
                    model: None,
                },
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(result.agent_path, "root/helper");
        assert_eq!(result.message, "Agent spawned successfully");

        // Verify the agent shows up in list
        let t2 = ListAgentsTool::new(rt);
        let list = t2.call_typed(ListAgentsArgs {}, &ctx).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].agent_path, "root/helper");
    }

    // ── spawn limit exceeded → Err (B5: no more fake-Ok) ──

    #[tokio::test]
    async fn test_spawn_agent_limit_exceeded() {
        let config = MultiAgentConfig {
            max_sub_agents: 1,
            ..MultiAgentConfig::enabled()
        };
        let rt = Arc::new(MultiAgentRuntime::new(
            config,
            Arc::new(StubClient),
            vec![],
            CancellationToken::new(),
            None,
            Language::En,
            None,
            None,
        ));

        let t = SpawnAgentTool::new(rt.clone(), std::env::current_dir().unwrap());
        let ctx = make_tool_ctx();
        let mk = |n: &str| SpawnAgentArgs {
            task_name: n.into(),
            task: "do the task".into(),
            fork_turns: None,
            model: None,
        };

        t.call_typed(mk("first"), &ctx)
            .await
            .expect("first spawn ok");
        // Second spawn hits the limit. The tool reports failure as an
        // Ok(SpawnAgentOutput) with an error message (not a typed Err).
        let result = t
            .call_typed(mk("second"), &ctx)
            .await
            .expect("second spawn call resolves");
        assert!(
            result.agent_path.is_empty()
                && result.message.contains("max agents reached"),
            "second spawn must report the limit failure, got: {result:?}"
        );
    }

    // ── create_all_tools: 4 active tools (followup_task and wait_agent
    //    dropped — §8.3 and the push-based result delivery) ──

    #[test]
    fn test_create_all_tools_returns_four() {
        let rt = make_runtime();
        let tools = create_all_tools(rt, std::env::current_dir().unwrap());
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"spawn_agent"));
        assert!(names.contains(&"send_message"));
        assert!(names.contains(&"list_agents"));
        assert!(names.contains(&"close_agent"));
        assert!(
            !names.contains(&"followup_task"),
            "deprecated followup_task must not be in the factory (§8.3)"
        );
        assert!(
            !names.contains(&"wait_agent"),
            "wait_agent must not be in the factory: results are pushed, not pulled"
        );
    }
}
