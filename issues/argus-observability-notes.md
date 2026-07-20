# How Argus Produces Its Observability Visualization

**Repo:** https://github.com/yessGlory17/argus

## TL;DR

Yes — Argus consumes the artifacts Claude Code already produces. It doesn't hook into
Claude Code at all; it's a pure passive consumer of the JSONL session transcripts that
Claude Code writes to disk.

It reads from `~/.claude/projects/` (in the home directory — the per-project
subdirectories like `-home-ffrujeri-nemo-gym/`, not a project-local `./.claude/projects`).

## Architecture

Argus is a VS Code extension with two halves:

1. **Session discovery** — an async filesystem scanner in the VS Code extension host
   recursively walks `~/.claude/projects/` to find session transcript files
   (`<session-uuid>.jsonl`).
2. **Live tailing** — a file watcher tails the active session's JSONL file, and a
   streaming parser processes new lines incrementally as Claude Code appends events.
   That's how the dashboard updates in "real time" without any integration on Claude
   Code's side.
3. **Parsing/analysis** — it extracts tool calls (Read, Edit, Bash, subagents, etc.),
   token usage per step, cache hit ratios, costs, retries, and compaction events from
   the transcript records, then runs a rule-based analyzer over them.
4. **Rendering** — a React 19 + Vite webview inside VS Code renders a seven-tab
   dashboard (Steps, Analysis, Cost, Performance, Flow, Context, Insights) using
   D3.js, Chart.js, and Recharts.

## What It Does *Not* Use

- No OpenTelemetry
- No Claude Code hooks or instrumentation
- No network egress — everything is derived locally; transcripts never leave the machine

## Practical Implications

- Argus can only show what's in the transcript. Anything Claude Code doesn't persist to
  the JSONL (e.g., internals of the API layer) is invisible to it.
- Its cost figures are recomputed from the token counts recorded in the transcript,
  rather than coming from any billing API.
