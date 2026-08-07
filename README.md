# phi-kernel-tools

[![crates.io](https://img.shields.io/crates/v/phi-kernel-tools.svg)](https://crates.io/crates/phi-kernel-tools)
[![Documentation](https://docs.rs/phi-kernel-tools/badge.svg)](https://docs.rs/phi-kernel-tools)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Kernel tools for phi-agent — LLM-callable Tool implementations injected via factory pattern.**

This crate provides the Tool implementations that the LLM uses to interact with `agent-works` infrastructure. It contains no infrastructure itself — only the `Tool` / `TypedTool` impls that bridge the LLM to the runtime.

## Architecture

```
agent-base          Runtime kernel (trait interfaces)
    ↑
agent-works         Infrastructure (MCP, Skills, Multi-Agent runtime)
    ↑
phi-kernel-tools    Tool implementations (this crate)
    ↑
phi-agent           Framework + CLI (consumer)
```

## Features

| Feature | Default | Tools |
|---------|---------|-------|
| `multi-agent` | ✅ | spawn_agent, send_message, followup_task, wait_agent, list_agents, close_agent |
| `skill` | ✅ | ApplySkillTool, SkillDetailTool |

## Installation

```toml
[dependencies]
phi-kernel-tools = "0.1.0"
```

Or pick specific features:

```toml
[dependencies]
phi-kernel-tools = { version = "0.1.0", default-features = false, features = ["multi-agent"] }
```

## Usage

```rust
use phi_kernel_tools::multi_agent;
use agent_works::{AgentBuilder, MultiAgentToolFactory};

// Create a factory from phi-kernel-tools
let factory: MultiAgentToolFactory =
    Arc::new(|rt| multi_agent::create_all_tools(rt));

// Inject into the builder
let runtime = AgentBuilder::new(client)
    .with_multi_agent(MultiAgentConfig::enabled())
    .with_multi_agent_tool_factory(factory)
    .build()
    .unwrap();
```

## License

MIT — see [LICENSE](LICENSE).
