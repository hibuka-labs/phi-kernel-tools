# Changelog

All notable changes to `phi-kernel-tools` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
