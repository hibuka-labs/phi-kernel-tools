//! Integration test: phi-kernel-tools → agent-works builder injection.
//!
//! Verifies that kernel tools can be injected into agent_works::AgentBuilder
//! via the factory pattern, and that the full pipeline works end-to-end.

#[cfg(feature = "multi-agent")]
use std::pin::Pin;
#[cfg(feature = "multi-agent")]
use std::sync::Arc;

#[cfg(feature = "multi-agent")]
use agent_base::{AgentResult, ChatMessage, LlmCapabilities, LlmClient, StreamChunk};
#[cfg(feature = "multi-agent")]
use agent_works::multi_agent::MultiAgentConfig;
#[cfg(feature = "multi-agent")]
use agent_works::{AgentBuilder, MultiAgentToolFactory};
#[cfg(feature = "multi-agent")]
use phi_kernel_tools::multi_agent;

// ── Minimal stub LLM client ──

#[cfg(feature = "multi-agent")]
struct StubClient;

#[cfg(feature = "multi-agent")]
#[async_trait::async_trait]
impl LlmClient for StubClient {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _tools: &[serde_json::Value],
        _reasoning: Option<&agent_base::ReasoningConfig>,
        _response_format: Option<&agent_base::ResponseFormat>,
    ) -> AgentResult<serde_json::Value> {
        Ok(serde_json::json!({"choices": [{"message": {"content": "ok"}}]}))
    }

    async fn chat_stream(
        &self,
        _messages: &[ChatMessage],
        _tools: &[serde_json::Value],
        _reasoning: Option<&agent_base::ReasoningConfig>,
        _response_format: Option<&agent_base::ResponseFormat>,
    ) -> AgentResult<Pin<Box<dyn futures_core::Stream<Item = AgentResult<StreamChunk>> + Send>>>
    {
        let chunks: Vec<AgentResult<StreamChunk>> = vec![
            Ok(StreamChunk::Text("ok".to_string())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        Ok(Box::pin(futures_util::stream::iter(chunks)))
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: false,
            max_context_tokens: None,
            max_output_tokens: None,
        }
    }
}

#[cfg(feature = "multi-agent")]
fn make_client() -> Arc<dyn agent_base::StreamClient> {
    agent_base::llm::adapt(Arc::new(StubClient))
}

/// Helper: get sorted list of registered tool names from a runtime.
#[cfg(feature = "multi-agent")]
fn registered_tool_names(runtime: &agent_base::AgentRuntime) -> Vec<String> {
    tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard.metadatas().into_iter().map(|m| m.name).collect()
    })
}

// ── End-to-end: create_all_tools + builder ──

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "multi-agent")]
async fn test_inject_multi_agent_tools_via_factory() {
    let client = make_client();
    let factory: MultiAgentToolFactory = Arc::new(multi_agent::create_all_tools);

    let runtime = AgentBuilder::new(client)
        .with_multi_agent(MultiAgentConfig::enabled())
        .with_multi_agent_tool_factory(factory)
        .build()
        .unwrap();

    let names = registered_tool_names(&runtime);
    assert!(
        names.contains(&"spawn_agent".to_string()),
        "spawn_agent should be registered"
    );
    assert!(
        names.contains(&"send_message".to_string()),
        "send_message should be registered"
    );
    assert!(
        names.contains(&"followup_task".to_string()),
        "followup_task should be registered"
    );
    assert!(
        names.contains(&"wait_agent".to_string()),
        "wait_agent should be registered"
    );
    assert!(
        names.contains(&"list_agents".to_string()),
        "list_agents should be registered"
    );
    assert!(
        names.contains(&"close_agent".to_string()),
        "close_agent should be registered"
    );
    assert_eq!(
        names.len(),
        6,
        "Expected exactly 6 multi-agent tools, got: {:?}",
        names
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "multi-agent")]
async fn test_builder_without_factory_has_no_multi_agent_tools() {
    let client = make_client();
    let runtime = AgentBuilder::new(client)
        .with_multi_agent(MultiAgentConfig::enabled())
        // No factory set
        .build()
        .unwrap();

    let names = registered_tool_names(&runtime);
    assert!(!names.contains(&"spawn_agent".to_string()));
    assert_eq!(names.len(), 0, "Expected 0 tools when no factory is set");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "multi-agent")]
async fn test_create_all_tools_returns_six_distinct_tools() {
    use agent_works::multi_agent::MultiAgentRuntime;
    use tokio_util::sync::CancellationToken;

    let cancel = CancellationToken::new();
    let rt = Arc::new(MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        make_client(),
        vec![],
        cancel,
        None,
        agent_base::Language::En,
        None,
        None,
    ));

    let tools = multi_agent::create_all_tools(rt);
    assert_eq!(tools.len(), 6);

    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"spawn_agent"));
    assert!(names.contains(&"send_message"));
    assert!(names.contains(&"followup_task"));
    assert!(names.contains(&"wait_agent"));
    assert!(names.contains(&"list_agents"));
    assert!(names.contains(&"close_agent"));
}

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "multi-agent")]
async fn test_multi_agent_disabled_does_not_register_tools() {
    let client = make_client();
    let factory: MultiAgentToolFactory = Arc::new(multi_agent::create_all_tools);

    // Default config has enabled=false
    let runtime = AgentBuilder::new(client)
        // No .with_multi_agent() call — stays disabled
        .with_multi_agent_tool_factory(factory)
        .build()
        .unwrap();

    let names = registered_tool_names(&runtime);
    assert!(!names.contains(&"spawn_agent".to_string()));
}
