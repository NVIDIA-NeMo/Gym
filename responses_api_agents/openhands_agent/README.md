# OpenHands Agent

Runs the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) headless inside a
benchmark's task sandbox, one conversation per agent session. The Environment Server (`single_agent_turn`)
seeds the session, activates it once through `/v1/responses`, closes it, and only then verifies.

This is not the harness behind `swe_agents`' `openhands` mode, which runs the nv-OpenHands fork of the
OpenHands application in SWE task containers with its own SWE verification. This adapter runs the upstream
SDK in any benchmark's task sandbox, and the benchmark's Resources Server verifies.

| | |
| --- | --- |
| Harness | `openhands-sdk` and `openhands-tools` 1.59.0, with every dependency at the version `uv.lock` pins for the Agent Server |
| Python in the sandbox | Standalone CPython 3.13.14 with the harness installed, built on the Agent Server host and uploaded |
| Execution | Agent sessions only, in a borrowed task sandbox or an agent-owned one (`sandbox_provider`). No host path, no `/run` |
| Model API | Chat Completions through the rollout-prefixed Gym Model Server URL |
| Tools | OpenHands' terminal, file editor, task tracker, `finish` and `think`. No Resources Server tool grants |
| Capability level | Evaluation. Observability records are returned (invocation, model-call references, tool calls with timing and status, context compactions), but harness conformance has not been run |
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
| `concurrency` | Activations this server runs at once; later ones wait. |
| `timeout` | Seconds one activation may run; a task's `metadata.agent_timeout_sec` can lower it. |
| `max_iterations` | OpenHands' `max_iteration_per_run`. Reaching it ends the run as `incomplete`, and the work is graded. |
| `condenser` | OpenHands' default `LLMSummarizingCondenser`. Its summaries use a copy of the agent's LLM, so they go to the same Model Server. |
| `system_prompt` | Text appended to OpenHands' own system prompt, before the request's `instructions` and system message. |
| `llm_settings` | OpenHands SDK `LLM` settings, limited to the names in `SUPPORTED_LLM_SETTINGS` in `app.py`. |
| `sandbox_provider`, `sandbox_config` | Only for an agent-owned sandbox, when Resources supplies no `sandbox_access`. `ttl_s` defaults to `sandbox_install_timeout_seconds + timeout + 600`. `metadata` is the provider block's `default_metadata`, then `sandbox_config.metadata`, then the adapter's `gym-agent`, `gym-session` and `gym-rollout` labels. |
| `network_access` | `inherit` (default), `model_only` or `model_and_tools`: egress of an agent-owned sandbox. `inherit` keeps the provider's network; the other two allow only the Model Server's host, because OpenHands keeps no tool grants. OpenSandbox only; a session that borrows a sandbox rejects any value other than `inherit`, as does one whose `sandbox_config.provider_options` sets its own `network_policy`. The Agent Server builds the runtime and uploads it, so a restricted sandbox needs no network for setup. |
| `sandbox_install_timeout_seconds` | Budget for runtime installation at seed. |
| `session_close_timeout_seconds` | Budget for each phase of close: stopping the harness, copying its artifacts, and releasing the sandbox. |

The adapter owns the model connection: the base URL, a placeholder API key (the Model Server holds the
provider credentials), the `agent` usage ID, and an `x-session-id` header set to the episode's capture key,
which is also the invocation ID. `llm_settings` cannot change these, and OpenHands silently ignores unknown
setting names, so the adapter refuses every name outside its list, along with settings that write files
(completion logs, tokenizer downloads), call other models (fallbacks), or stream. OpenHands' own model retries
are off (`num_retries: 0`), and its retry settings are refused: through LiteLLM 1.105.7 it cannot tell a 4xx from a
transient failure, since LiteLLM raises any error whose text mentions a rate limit as a `RateLimitError`, and OpenHands
also retries HTTP 409. The Model Server retries transient failures. `api_mode` defaults to
`chat`, because OpenHands' `auto` mode sends models its metadata marks as Responses API models (such as
gpt-5) to `/responses`. The shipped config sets `reasoning_effort: null`, because OpenHands otherwise sends
`reasoning_effort: high` to models it believes support it; keep sampling and reasoning controls on the Model
Server.

## Requests

An activation accepts one user text message with an optional leading system message, `instructions`,
`model` (the configured one), and `metadata` that is empty or holds only `agent_timeout_sec`. Every other
request field, including `temperature`, `top_p`, `max_output_tokens` and `tools`, is refused with HTTP 422.

## Session lifecycle

**Seed** refuses, with a final 422, a required tool grant (OpenHands uses its own tools), a non-direct
sandbox connection, a seed with neither `sandbox_access` nor `sandbox_provider`, an invalid
`sandbox_provider` or `sandbox_config`, a workdir that is relative or overlaps the adapter's own directories,
and a loopback Model Server URL for a remote sandbox (start the servers with `++use_absolute_ip=true`).
Optional grants are ignored. Seed then installs the runtime, or reuses a healthy one, and uploads the runner.

**Runtime installation** uses Gym's `SandboxPythonRuntime`. The Agent Server host builds the runtime with `uv`:
a pinned python-build-standalone CPython 3.13.14 and the OpenHands packages for the sandbox's platform, at the
versions installed in the Agent Server's own environment. The sandbox needs no network, no `uv` and no Python
of its own, only Linux x86_64 or aarch64 with glibc 2.28 or later, bash, tar with gzip, and a writable `/tmp`.
The first build for each glibc level takes minutes and is cached on the host under Gym's `cache_dir`; later
seeds upload the cached archives. The runtime is published atomically in
`/tmp/nemo-gym-openhands-runtime-<key>/`, keyed by everything that goes into it, and reused by later seeds in
the same sandbox. The task image's Python is never used or changed.

**Activation** runs `sandbox_runner.py` (uploaded as `openhands_runner`) under Gym's process supervisor in
the task's workdir. The interpreter runs through an `openhands-gym` link beside the runtime's `python/`
directory, so `pkill python3` or `pkill -f python` in a task cannot stop it. The runner builds the agent the
way OpenHands' default preset does (terminal, file editor, task tracker, default condenser, CLI-mode system
prompt) with three deliberate changes: the terminal uses the subprocess backend, because the tmux backend
starts a tmux server that daemonizes out of the supervised process group; bash history goes to the session
directory; and bracketed paste is turned off (see below). It appends each OpenHands event to
`events.jsonl` as it happens and writes `result.json` at the end. The supervisor's SIGTERM, at the deadline
or on close, makes the runner write its result and close the terminal before it exits.

**Close** confirms that the harness stopped, copies the artifacts once, removes the session directory and
disconnects. A borrowed sandbox stays with Resources; an owned one is stopped. Close always returns an
observation bundle, with an `agent_activation_interrupted` gap when no activation produced one.

## Files written in the sandbox

- `/tmp/nemo-gym-openhands-runtime-<key>/`: the shared runtime and its `openhands-gym` link (kept after close).
- `/tmp/nemo-gym-openhands-sessions/<id>/`: runner input, events, result, logs, OpenHands' conversation
  state, task-tracker file and bash history (removed on close).
- OpenHands' per-user directory (`~/.openhands`) is moved into the session directory with
  `OH_PERSISTENCE_DIR`; nothing is written under the task's home. Tool shells keep the task's `HOME` and
  `XDG_*` directories. They also see `OH_PERSISTENCE_DIR`, `OPENHANDS_SUPPRESS_BANNER`, OpenHands' own
  `AI_AGENT=openhands`, `HISTFILE` and `INPUTRC`. `INPUTRC` points at a session file that includes the
  readline file bash would have read (`~/.inputrc` or `/etc/inputrc`) and turns bracketed paste off: at 1.59.0
  the subprocess terminal otherwise leaves bash 5.1+'s `ESC[?2004h` codes in the tool output.

## Responses and outcomes

The response holds the agent's messages, reasoning, tool calls and tool results in order. Each tool
result is the tool message the model read: OpenHands adds its own notes to a tool's output, such as the
exit code, the working directory, a no-output timeout or truncation, and the runner records that text.
OpenHands delivers its final answer through the `finish` tool; that answer is also added as an assistant
message.
Status follows OpenHands' own rule at 1.59.0: `conversation.run()` raises only for errors that end the run.

| OpenHands outcome | Response | Verified |
| --- | --- | --- |
| `finished` | `completed` | Yes |
| `MaxIterationsReached`, `MaxBudgetReached`, stuck detection, a rollout stop, or Gym's deadline after a model call returned | `incomplete`; `stop_reason` names it | Yes |
| Context window exceeded | `incomplete`, `stop_reason` `context_limit` | Yes |
| Gym's deadline before any model call returned | `failed`, `stop_reason` `timeout`, with `ng_agent_outcome` `unknown` | No: the outcome policy retries it by default |
| A model error | `failed`, with `ng_agent_outcome` `model_request_error` and the HTTP status the provider's error recorded, or `model_transport_error` | By the outcome policy: by default 400, 413 and 422 are verified, `model_transport_error` and 408, 425, 429 and 5xx are retried, and other statuses fail the episode |
| Any other error that ends the run, including a model error that recorded no HTTP status | `failed`, with `ng_agent_outcome` `unknown` | No: the outcome policy retries it by default |
| No result, unconfirmed termination, or a sandbox failure | HTTP error from `/v1/responses` | No |

Gym's deadline before any model reply leaves nothing to grade, and the adapter cannot tell its cause: the
runner may still have been starting, or the Model Server may not have answered. The Environment Server's
`agent_outcome_policy` retries `unknown` by default; `by_failure_kind` changes that. Once a model call has
returned, the deadline is graded with the work done so far, even if the next call was still waiting on the
Model Server: the adapter cannot tell a slow reply from a Model Server that stopped answering. A rollout stop
is graded either way.

The runner records the error that ended a run as its class and the classes it was raised from
(`raise ... from`), each with any HTTP status a provider's error carried. An error's text can hold a
provider's response body, so it stays out of the response; the runner writes the traceback to its stderr,
and the adapter logs the end of that stderr when a run fails.

Tool errors are returned to the model and do not end the run. A terminal command that exits nonzero is
recorded as a `failed` tool call. Usage is reported only when every model response the conversation used,
including condenser summaries, has a usage record; otherwise it stays unknown.
Cached and reasoning token details stay unknown (`null`), with observation gaps: OpenHands SDK 1.59.0
normalizes absent details to zero, so its counters cannot establish complete totals. The raw SDK stats,
including any positive detail counters, remain in the run result artifact.

## Observations

The bundle holds an `AgentInvocation` (status, stop reason, duration, and references to the agent's own
model calls from OpenHands' per-call usage records and response IDs), a `ToolCallObservation` for each tool
call (timed from when the runner saw the action and its result), a `ContextCompactionObservation` for each
condensation, which references the condenser's summary call, and a `SandboxObservation`. The condenser's
calls carry the same `x-session-id` header as the agent's, so the Model Server captures them under this
invocation. Gaps name what the events do not show, such as token counts around a compaction.

## Limits

- Resources Server tool grants (MCP or direct HTTP) are not supported; required grants are refused.
- `network_access` restricts only an agent-owned OpenSandbox sandbox. With the default `inherit`, an owned
  sandbox's egress is open unless its provider limits it. A borrowed sandbox keeps the network its Resources
  Server gave it.
- The first seed for each glibc level builds the runtime on the Agent Server host, which needs `uv` and
  access to PyPI and GitHub (python-build-standalone) or mirrors of them.
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
SIGTERM during the run. `tests/test_sandbox.py` stages an activation through Gym's harness runner with that
stand-in and covers rollout stops before launch and before the first event.
