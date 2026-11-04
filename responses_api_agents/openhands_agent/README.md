# OpenHands Agent

Runs the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) headless inside a
benchmark's task sandbox, one conversation per agent session. The Environment Server (`single_agent_turn`)
seeds the session, activates it once through `/v1/responses`, closes it, and only then verifies.

| | |
| --- | --- |
| Harness | `openhands-sdk` and `openhands-tools` 1.59.0, every dependency hash-locked in `sandbox_requirements.txt` |
| Python in the sandbox | Standalone CPython 3.13.14 (python-build-standalone 20260805), verified on the Agent Server and uploaded |
| Execution | Agent sessions only, in a borrowed task sandbox or an agent-owned one (`sandbox_provider`). No host path, no `/run` |
| Model API | Chat Completions through the rollout-prefixed Gym Model Server URL |
| Tools | OpenHands' terminal, file editor, task tracker, `finish` and `think`. No Resources Server tool grants |
| Capability level | Evaluation, with Observability records (invocation, model-call references, tool calls with timing and status, context compactions) |
| Token IDs | Not captured; evaluation only |

## Configuration

`configs/openhands_agent.yaml` is the session config; a run config binds it to a benchmark:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/openhands_agent/configs/openhands_agent.yaml
  - environment_servers/single_agent_turn/configs/single_agent_turn.yaml

environment_routing_mode: taskset
environment_server_routes:
  terminal_bench_2_1: single_agent_turn

single_agent_turn:
  environment_servers:
    single_agent_turn:
      resources_server:
        name: terminal_bench_2_1_resources_server
      agent_server:
        name: openhands_agent
      resources_tool_transports: []
```

Terminal-Bench 2.1 does not list `openhands_agent` in `allowed_agents` yet; start the servers with
`--allow-unsupported-pairing` until it does.

| Field | Meaning |
| --- | --- |
| `model` | Model name sent to the Model Server. OpenHands addresses it as `openai/<model>`. |
| `timeout` | Seconds one activation may run; a task's `metadata.agent_timeout_sec` can lower it. |
| `max_iterations` | OpenHands' `max_iteration_per_run`. Reaching it ends the run as `incomplete`, and the work is graded. |
| `condenser` | OpenHands' default `LLMSummarizingCondenser`. Its summaries use a copy of the agent's LLM, so they go to the same Model Server. |
| `system_prompt` | Text appended to OpenHands' own system prompt, before the request's `instructions` and system message. |
| `llm_settings` | OpenHands SDK `LLM` settings, limited to the names in `SUPPORTED_LLM_SETTINGS` in `app.py`. |
| `sandbox_provider`, `sandbox_config` | Only for an agent-owned sandbox, when Resources supplies no `sandbox_access`. |
| `sandbox_install_timeout_seconds` | Budget for runtime installation at seed. |

The adapter owns the model connection: the base URL, a placeholder API key (the Model Server holds the
provider credentials), the `agent` usage ID, and an `x-session-id` header set to the episode's capture key,
which is also the invocation ID. `llm_settings` cannot change these, and OpenHands silently ignores unknown
setting names, so the adapter refuses every name outside its list, along with settings that write files
(completion logs, tokenizer downloads), call other models (fallbacks), or stream. The shipped config sets
`reasoning_effort: null`, because OpenHands otherwise sends `reasoning_effort: high` to models it believes
support it; keep sampling and reasoning controls on the Model Server.

## Requests

An activation accepts one user text message with an optional leading system message, `instructions`,
`model` (the configured one), and `metadata.agent_timeout_sec`. Every other request field, including
`temperature`, `top_p`, `max_output_tokens` and `tools`, is refused with HTTP 422.

## Session lifecycle

**Seed** refuses, with a final 422, a required tool grant (OpenHands uses its own tools), a non-direct
sandbox connection, a seed with neither `sandbox_access` nor `sandbox_provider`, a missing `model_server`,
a workdir that is relative or overlaps the adapter's own directories, and a loopback Model Server URL for a
remote sandbox (start the servers with `++use_absolute_ip=true`). Optional grants are ignored. Seed then
installs the runtime, or reuses a healthy one, and uploads the runner.

**Runtime installation** (`install_openhands_runtime.sh`) needs Linux x86_64 or aarch64 with glibc, bash,
tar, gzip, sha256sum, flock, tee, `mv -T`, a writable `/tmp`, and access to PyPI: pip downloads the locked
packages (about 1.1 GB installed). A cold install took 32 s on a Slurm compute node. The runtime is keyed by a
fingerprint of the Python build, the installer and the lock file, published atomically under
`/tmp/nemo-gym-openhands-runtime/`, and reused by later seeds in the same sandbox. The task image's Python is
never used or changed.

**Activation** runs `sandbox_runner.py` (uploaded as `openhands_runner`) under Gym's process supervisor in
the task's workdir. The interpreter runs as `openhands-gym`, so `pkill python3` or `pkill -f python` in a task
cannot stop it. The runner builds the agent the way OpenHands' default preset does (terminal, file editor,
task tracker, default condenser, CLI-mode system prompt) with two deliberate changes: the terminal uses the
subprocess backend, because the tmux backend starts a tmux server that daemonizes out of the supervised
process group; bash history goes to the session directory; and bracketed paste is turned off (see below). It appends each OpenHands event to
`events.jsonl` as it happens and writes `result.json` at the end. The supervisor's SIGTERM, at the deadline
or on close, makes the runner write its result and close the terminal before it exits.

**Close** confirms that the harness stopped, copies the artifacts once, removes the session directory and
disconnects. A borrowed sandbox stays with Resources; an owned one is stopped. Close always returns an
observation bundle, with an `agent_activation_interrupted` gap when no activation produced one.

## Files written in the sandbox

- `/tmp/nemo-gym-openhands-runtime/`: the shared runtime (kept after close).
- `/tmp/nemo-gym-openhands-sessions/<id>/`: runner input, events, result, logs, OpenHands' conversation
  state, task-tracker file and bash history (removed on close).
- OpenHands' per-user directory (`~/.openhands`) is moved into the session directory with
  `OH_PERSISTENCE_DIR`; nothing is written under the task's home. Tool shells keep the task's `HOME` and
  `XDG_*` directories. They also see `OH_PERSISTENCE_DIR`, `OPENHANDS_SUPPRESS_BANNER`, OpenHands' own
  `AI_AGENT=openhands`, `HISTFILE` and `INPUTRC`. `INPUTRC` points at a session file that includes the
  readline file bash would have read (`~/.inputrc` or `/etc/inputrc`) and turns bracketed paste off: at 1.59.0
  the subprocess terminal otherwise leaves bash 5.1+'s `ESC[?2004h` codes in the tool output.

## Responses and outcomes

The response holds the agent's messages, reasoning, tool calls and tool results in order. OpenHands
delivers its final answer through the `finish` tool; that answer is also added as an assistant message.
Status follows OpenHands' own rule at 1.59.0: `conversation.run()` raises only for errors that end the run.

| OpenHands outcome | Response |
| --- | --- |
| `finished` | `completed` |
| `MaxIterationsReached`, `MaxBudgetReached`, stuck detection, the deadline, or a stop request | `incomplete`, graded |
| Context window exceeded | `incomplete`, graded, outcome `model_context_limit` |
| A model error after OpenHands' own retries | `failed`, outcome `model_request_error` with the HTTP status, or `model_transport_error` |
| Any other error that ends the run | `failed`, outcome `unknown` (retryable) |
| No result, unconfirmed termination, or a sandbox failure | HTTP error from `/v1/responses` |

Tool errors are returned to the model and do not end the run. A terminal command that exits nonzero is
recorded as a `failed` tool call. Usage is reported only when every model response the conversation used,
including condenser summaries, has a usage record; otherwise it stays unknown.

## Observations

The bundle holds an `AgentInvocation` (status, stop reason, duration, model-call references from OpenHands'
per-call usage records and response IDs), a `ToolCallObservation` for each tool call (timed from when the
runner saw the action and its result), a `ContextCompactionObservation` for each condensation, and a
`SandboxObservation`. Gaps name what the events do not show, such as token counts around a compaction.

## Limits

- Resources Server tool grants (MCP or direct HTTP) are not supported; required grants are refused.
- The sandbox needs PyPI access at seed; there is no offline install yet.
- Tool commands run as the task's user and can see the runtime. In the TB2.1 smoke, the agent found the
  runtime's Python with `find / -name "python*"` in an image without Python and used it to test its answer.
  The task's verifier never uses it, but the agent had an interpreter the task image does not provide.
- Token IDs are not captured, so the adapter is evaluation only.
- Browser tools and sub-agents are not enabled.
- `prompt_cache_key` is sent as OpenHands sends it; Gym's Chat Completions schema accepts it.

## Tests

```bash
gym env test +entrypoint=responses_api_agents/openhands_agent
```

`tests/test_sandbox_runner.py` runs the real runner process against a stand-in for the SDK, including a
SIGTERM during the run.
