# Changelog

All notable changes to `phi-kernel-tools` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-09-27

### Added
- **Per-call `timeout_ms` on `execute_command`** (`0` = never kill): background
  jobs default to indefinite (daemon-style servers/watchers), foreground keeps
  the tool-level fuse; an explicit `timeout_ms` arms a per-call kill fuse.
  Registry entries and snapshots carry `timeout_ms` so the TUI can tell bounded
  jobs from daemons. Tool description rewritten around the two task shapes.
- **`spawn_agent` `tools` parameter** (ReadOnly / Write / researcher / coder /
  reviewer / tester presets; default read-only) with capability echo (label +
  actually-registered tool set, `degraded` reason when degraded); `list_agents`
  exposes `spawned_tools`.
- Spawn echo marks recycled predecessors ("recycled a finished agent with the
  same path"); `task_output` gains a scope fence plus a teaching not-found error
  for sub-agent ids (reports are pushed, not polled).

### Changed
- Bump `agent-base` to 0.8.0, `agent-works` to 0.9.0.

## [0.8.0] - 2026-09-18

### Added
- **Generic task registry framework**: reusable task lifecycle tracking with
  typed metadata, status derivation, and snapshot support.
- **`force` flag on `close_agent`**: prevents accidental force-kill of
  long-running child agents. Callers must set `force=true` to explicitly
  acknowledge partial-work loss when closing a running agent.

## [0.7.0] - 2026-09-11

### Added
- **Context-rotation tools**: `notes_write`, `notes_read`, `handoff_write` —
  file-backed note and handoff tools for context-rotation workflows.
- Tool descriptions aligned with Codex style conventions.

### Changed
- Tool descriptions refactored for clarity and consistency.

## [0.6.0] - 2026-09-06

### Added
- Push-model multi-agent tool surface with child path discipline.
- `spawn_agent` gains a `task` field with Focus-based prompt expansion; task
  text is capped at 3-5 sentences with a scaffolded report format.
- `list_agents` exposes queued status, running seconds, and a `delivery_note`
  describing how child results will reach the parent; pre-close warnings are
  emitted when children are still pending.
- File tools return `edit_lines` / `write_mode` metadata so TUIs can render
  diffs.

### Fixed
- Spawn task-delivery failures are reported and the orchestration is closed
  cleanly instead of hanging.
