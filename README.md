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

All features are opt-in. Use `full` to enable all.

| Feature | Tools |
|---------|-------|
| `file` | `read_file`, `write_file`, `list_files`, `edit_file` |
| `multi-agent` | `spawn_agent`, `send_message`, `followup_task`, `wait_agent`, `list_agents`, `close_agent` |
| `shell` | `execute_command` |
| `full` | Enables all of the above |

## Installation

```toml
[dependencies]
phi-kernel-tools = "0.1.1"
```

Or pick specific features:

```toml
[dependencies]
phi-kernel-tools = { version = "0.1.1", features = ["file", "shell"] }
```

## Usage

### File Tools

```rust
use phi_kernel_tools::file::{ReadFileTool, WriteFileTool, ListFilesTool, EditFileTool};
use std::path::PathBuf;

let cwd = PathBuf::from(".");

builder
    .register_tool(ReadFileTool::new(cwd.clone()))
    .register_tool(WriteFileTool::new(cwd.clone()))
    .register_tool(ListFilesTool::new(cwd.clone()))
    .register_tool(EditFileTool::new(cwd.clone()));
```

`EditFileTool` supports precision text replacement with a 4-level fallback matching strategy (exact match → rstrip → trim → Unicode NFC normalization).

### Shell Tools

```rust
use phi_kernel_tools::local_shell::LocalShellTool;

builder.register_tool(LocalShellTool::new(30_000));  // 30s timeout
```

### Multi-Agent Tools

```rust
use phi_kernel_tools::multi_agent;
use agent_works::{MultiAgentRuntime, MultiAgentConfig};

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

The `spawn_agent` tool supports these parameters:

| Parameter | Description |
|-----------|-------------|
| `fork_history` | `"none"` (default), `"all"`, or N — how much parent history the child inherits |
| `depth` | Nesting depth (1 = direct child, 2 = grandchild, etc.) |
| `model` | Optional model override for the sub-agent |
| `reasoning_effort` | Optional reasoning effort (`low` / `medium` / `high`) |
| `agent_type` | Optional role type that maps to a preset config |
| `system_prompt` | Optional custom system prompt (overrides `agent_type`) |

## License

MIT — see [LICENSE](LICENSE).
