# Don't add new inconsistencies (item 16)

This page summarizes the harness consistency plan. It lists differences between adapters that look accidental, and a seven-step plan to share the code.

**Status:**

- **Accepted:** step 1, the outcome ladder (#5870), with all six of its decisions, including decision 6 on retries.
- **Proposed:** steps 2-7. Raise departures from them as suggestions, not blockers.

## Don't let a PR add a new instance of these

| Step | Pattern to flag | Shared target |
|---|---|---|
| 1 (accepted) | Its own status logic; a bare 502/503; a context limit as `failed`; a harness retry loop that retries 4xx | `RunEvidence` → `finish_response` (#5870) |
| 2 | Building the model URL or `x-session-id` by hand; a partial usage sum | A shared `ModelConnection`; the usage rule |
| 3 | A new CLI installer that downloads inside the sandbox, checks no digest, or has no lock or health check | A shared CLI runtime installer, modelled on Droid's (#5537) |
| 4 | Its own workdir, user and env rules; plain `HTTPException(422)`; no `ttl_s` default; new label keys; a missing `calls_model_from_sandbox` | `sandbox_seed.py` helpers; labels `gym-agent`, `gym-session`, `gym-rollout` |
| 5 | A runner `*.py` uploaded into the shared runtime directory; `input.json` kept after reading; a new place to read the time budget | Session-dir runner without `.py`, an interpreter link, `-E -s -B`; `resolve_agent_timeout` |
| 6 | Its own `_rollout_sandbox_session` override or close-reply code; a close reply missing `cleanup_confirmed` | `SandboxSessionState`; `sandbox_close_response()` |
| 7 | A new client-specific field fix in `nemo_gym/core/responses_api_types.py`; a second adapter for an existing harness; an untested version pin | A client replay registry; one adapter and one tested pin per harness |

## Kept different on purpose: don't flag

These differences come from the upstream runtimes:

- model API: Anthropic Messages, Responses, Chat Completions;
- packaging: Python SDK, Node CLI, single binary;
- the state-directory variable (`CODEX_HOME`, `PI_CODING_AGENT_DIR`, …);
- error evidence: structured status or text only;
- upstream quirks: OpenClaw's respawn, Codex's 429 handling, Copilot's secret masking, Terminus' tmux, mini-SWE's bash-only tools, Pool's EULA.

## Comparison page

The Gym Harness Comparison artifact (https://claude.ai/artifact/7W8SAwRnjLC26xeXCHJgJR) has the current per-harness matrix. Use it to see how a harness differs from the others before calling a difference new.
