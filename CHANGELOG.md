# Changelog

All notable changes to `phi-kernel-tools` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.0] - 2026-11-14

### Added
- **The terminal stops being a black box.** A PTY-backed task registry
  (`pty_task`), a term session that owns a real pty (`term_session`), and an
  interactive-prompt detector (`interaction/`) that turns "the command is
  stuck on a Y/N prompt" into something the model can *see* — the detector is
  a pure, policy-free matcher over output tails that reports the prompt's
  shape (password / ssh host key / pager / confirm / menu / press-enter) and
  never decides what to send.
- **`task_input` — send an answer to a running task.** The model reads the
  screen, decides what to answer, and sends it here. Three policy tiers
  govern the send: `Auto` (deterministic safe answer, straight through),
  `Cache` (first send per action key asks once, later sends go direct), and
  `Stop` (the hard boundary — password, 2FA, ssh host key, unknown patterns:
  **zero bytes leave the tool** and it returns `[needs user]`). The consent
  channel is `agent_base`'s `InteractionChannel` seam; with no channel
  mounted, a cache-tier send fails closed rather than answering for the user.
- **`local_shell` reports a blocked prompt instead of hanging.** A shell
  command that waits on input now surfaces the prompt as a notice the model
  can act on.

### Changed
- **Breaking (dependency):** `agent-base` requirement bumped from 0.13.0 to
  0.14.0 and `agent-works` from 0.13.0 to 0.14.0, keeping the framework on a
  single version of each family crate alongside phi-agent 0.22.0.



## [0.13.0] - 2026-11-09

### Added
- **`read_image` — one entry point for a picture the model should look at**,
  whether it lives on disk or behind an http(s) URL. The result is the
  vision channel's native shape: a metadata text line (dimensions, format,
  size) plus the bytes as an image content. Type is sniffed from magic bytes
  (an HTML page claiming to be a PNG is refused), and the URL fetch refuses
  anything that is not a public address.
- **`spill` — oversized output goes to the artifact store** instead of
  ending in a truncation marker; the artifact is fetched back exactly like a
  code-cell artifact.

### Changed
- **Breaking (dependency):** `agent-base` 0.13.0 and `agent-works` 0.13.0
  (the image budget and the artifact store live upstream).
- **`read_file` takes `paths: [...]`** for batched multi-file reads.
- **A session-scoped store can be re-pointed**, so a resumed session writes
  its audit trail into its own tree instead of the bootstrap session's.

### Fixed
- **`list_files` says what it returns** — the per-file sizes are named in the
  tool description.

## [0.11.0] - 2026-10-24

### Added
- **`~` expands to the home directory in the file tools** — `~/…` and bare
  `~` resolve to the home directory (docs and skill files spell paths that
  way; the model was wasting tool calls re-finding `~/.phimint/skills/…`
  by hand). `~user` is deliberately not touched — only the `~` the writer
  meant as "my home" is unambiguous.

### Changed
- Bump `agent-base` to 0.11.0 and `agent-works` to 0.11.1 (per-call timeout
  budget, awaiting-human park).

## [0.10.0] - 2026-10-18

### Added
- **Opt-in path confinement for the file tools** — restrict file reads/writes
  to a configured root.
- **Context rotation keeps the lossless `ChatMessage`** in `MessageRecord`, so
  rotated history can be rebuilt without re-parsing display text.

### Fixed
- Local shell decodes child pipe output lossily: invalid UTF-8 no longer kills
  the capture pipe (the forwarder treated strict line decoding EOF as EPIPE).
- Background shell slices the bounded output buffer at char boundaries — no
  more panic/dropped tail when the cut lands mid-CJK.

## [0.9.1] - 2026-10-07

### Fixed
- `read_item` offsets count characters, not bytes (context rotation)
- Background-task output stays reachable until consumed (#35)

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
