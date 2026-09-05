//! Integration test: phi-kernel-tools → agent-works builder injection.
//!
//! Verifies that kernel tools can be injected into agent_works::AgentBuilder
//! via the factory pattern, and that the full pipeline works end-to-end.

#[cfg(feature = "multi-agent")]
use std::sync::Arc;

#[cfg(feature = "multi-agent")]
use agent_base::StreamChunk;
#[cfg(feature = "multi-agent")]
use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
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
impl LlmProvider for StubClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks = vec![
            Ok(StreamChunk::Text("ok".to_string())),
            Ok(StreamChunk::Stop {
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

#[cfg(feature = "multi-agent")]
fn make_client() -> Arc<dyn LlmProvider> {
    Arc::new(StubClient)
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
    let factory: MultiAgentToolFactory =
        Arc::new(move |rt| multi_agent::create_all_tools(rt, std::env::current_dir().unwrap()));

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
        names.contains(&"list_agents".to_string()),
        "list_agents should be registered"
    );
    assert!(
        names.contains(&"close_agent".to_string()),
        "close_agent should be registered"
    );
    // followup_task is deprecated and no longer registered (§8.3); its
    // trigger semantics now live in send_message(trigger=true).
    assert!(
        !names.contains(&"followup_task".to_string()),
        "followup_task must not be registered anymore"
    );
    // wait_agent is gone too: results are pushed to the parent automatically;
    // the parent waits by ending its turn, never by pulling with a tool.
    assert!(
        !names.contains(&"wait_agent".to_string()),
        "wait_agent must not be registered anymore"
    );
    assert_eq!(
        names.len(),
        4,
        "Expected exactly 4 multi-agent tools, got: {:?}",
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
async fn test_create_all_tools_returns_four_distinct_tools() {
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

    let tools = multi_agent::create_all_tools(rt, std::env::current_dir().unwrap());
    assert_eq!(tools.len(), 4);

    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"spawn_agent"));
    assert!(names.contains(&"send_message"));
    assert!(names.contains(&"list_agents"));
    assert!(names.contains(&"close_agent"));
}

/// Compat red line (§12 stage 3): the exact wire format phimint's decompose
/// prompt instructs — `spawn_agent task_name=<name>, message="..."` — must
/// still spawn successfully through the untyped `Tool::call` path (which is
/// what the LLM actually drives: serde aliases on the real dispatch route).
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "multi-agent")]
async fn phimint_legacy_wire_shape_spawns_end_to_end() {
    use agent_works::multi_agent::MultiAgentRuntime;
    use tokio_util::sync::CancellationToken;

    let rt = Arc::new(MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        make_client(),
        vec![],
        CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ));

    let tools = multi_agent::create_all_tools(rt.clone(), std::env::current_dir().unwrap());
    let spawn = tools
        .iter()
        .find(|t| t.name() == "spawn_agent")
        .expect("spawn_agent in factory");

    // Byte-for-byte the new minimal wire shape: task_name + task carry
    // everything the child needs (no conversation context is inherited
    // unless fork_turns is set).
    let wire = serde_json::json!({
        "task_name": "slice_1",
        "task": "Context: repo layout\nInvestigate and report: module boundaries"
    });
    let ctx = agent_base::ToolContext::for_test();
    let content = spawn
        .call(&wire, &ctx)
        .await
        .expect("minimal wire shape must not error");
    let text = agent_base::tool::content_text(&content);
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["agent_path"], "root/slice_1");
    assert_eq!(v["message"], "Agent spawned successfully");

    // The child really works: wait collects its answer.
    let res = rt.wait_for_result(Some("root/slice_1"), 5000).await;
    assert_eq!(res.status, "ok");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "multi-agent")]
async fn test_multi_agent_disabled_does_not_register_tools() {
    let client = make_client();
    let factory: MultiAgentToolFactory =
        Arc::new(move |rt| multi_agent::create_all_tools(rt, std::env::current_dir().unwrap()));

    // Default config has enabled=false
    let runtime = AgentBuilder::new(client)
        // No .with_multi_agent() call — stays disabled
        .with_multi_agent_tool_factory(factory)
        .build()
        .unwrap();

    let names = registered_tool_names(&runtime);
    assert!(!names.contains(&"spawn_agent".to_string()));
}
