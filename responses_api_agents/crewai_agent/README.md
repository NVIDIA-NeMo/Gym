# CrewAI Agent

This adapter runs a [CrewAI](https://github.com/crewAIInc/crewAI) crew as a Gym harness, one crew turn per agent
session. CrewAI is a Python framework, so the crew runs in the Agent Server process. Its model is the Gym Model
Server, and its tools run where the episode puts them: a shell in the episode's task sandbox, and the tools of the
episode's MCP grants. The sandbox is the one the Resources Server lends the session, which the adapter never stops,
or, when it lends none and `sandbox_provider` is set, one the adapter starts and stops itself.

- Harness: `crewai==1.15.34`, pinned in [`pyproject.toml`](pyproject.toml) and locked in [`uv.lock`](uv.lock).
- Execution: in process, under an Environment Server such as `single_agent_turn`. There is no `/run` route and no
  host path: `/v1/responses` without a seeded agent session answers 409.
- Capability level: Observability, plus Environment tools and task sandboxes. The close bundle carries an invocation
  record with every model-call reference, tool-call records with executor timing, and the agent's sandbox record. The
  [harness conformance probes](../../scripts/harness_conformance/README.md) for `crewai` meet P0 in all eight
  scenarios.
- Training: not qualified. `token_id_capture` is untested.

## Configure a pairing

[`configs/crewai_agent.yaml`](configs/crewai_agent.yaml) declares the agent `crewai_agent` with
`resources_server: null`, so the Environment Server binds the benchmark. Keep `num_workers: 1`: sessions live in the
process that seeded them.

### Terminal-Bench 2.1, with the shell in the task sandbox

Prepare the data as the [benchmark README](../../benchmarks/terminal_bench_2_1/README.md) describes, and save this as
`run.yaml`:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/crewai_agent/configs/crewai_agent.yaml
  - environment_servers/single_agent_turn/configs/single_agent_turn.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml
  - nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

environment_routing_mode: taskset
environment_server_routes:
  terminal_bench_2_1: single_agent_turn

single_agent_turn:
  environment_servers:
    single_agent_turn:
      resources_server:
        name: terminal_bench_2_1_resources_server
      agent_server:
        name: crewai_agent
      resources_tool_transports: []

crewai_agent:
  responses_api_agents:
    crewai_agent:
      timeout: 900
      context_window_size: 131072
      shell_timeout_seconds: 300
```

Put `policy_base_url`, `policy_api_key` and `policy_model_name` in `env.yaml`, and export `OPENSANDBOX_DOMAIN` and
`OPENSANDBOX_API_KEY`. The crew calls the Model Server from the Agent Server, so the task sandbox never needs to
reach Gym: `use_absolute_ip` is not required. The benchmark declares `crewai_agent` in its `allowed_agents`.
Materialized task rows take no request overrides, so put a per-call output cap on the Model Server
(`sampling_overrides: {max_tokens: 8192}`) rather than on `gym eval run`:

```bash
gym env start --config run.yaml
gym eval run --no-serve --config run.yaml \
  --input results/tb21-input.jsonl --output results/crewai-tb21/rollouts.jsonl --concurrency 1
```

### A Resources Server's tools, as an MCP grant

Swapping the benchmark changes configuration only. This composition runs
[`example_single_tool_call`](../../resources_servers/example_single_tool_call/README.md) with its tools exposed over
MCP, and no sandbox:

```yaml
config_paths:
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml
  - responses_api_agents/crewai_agent/configs/crewai_agent.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml

example_single_tool_call:
  resources_servers:
    example_single_tool_call:
      entrypoint: app.py
      domain: agent
      verified: false
      expose_tools_over_mcp: true

example_single_tool_call_single_agent_turn:
  _inherit_from: single_agent_turn_legacy
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        type: resources_servers
        name: example_single_tool_call
      agent_server:
        type: responses_api_agents
        name: crewai_agent
      resources_tool_transports: [mcp]
```

```bash
gym eval run --no-serve --config run.yaml --agent crewai_agent \
  --input resources_servers/example_single_tool_call/data/example.jsonl --output results/crewai-mcp/rollouts.jsonl
```

## How a session runs

| Call | What the adapter does |
| --- | --- |
| Seed (`/v1/agent_sessions`) | Lists the tools of each MCP grant, then connects to the lent sandbox, or starts its own with `sandbox_config`, and creates the shell's session directory `/tmp/nemo-gym-crewai-sessions/<id>`, outside the task directory. |
| Activation (`/ng-rollout/<capture key>/v1/responses`) | Builds one CrewAI agent and one task from the request and runs `Crew.akickoff()` once. Identical retries join the same run; a different request answers 409. |
| Close (`/v1/agent_sessions/close`) | Cancels the crew's calls in flight and fails its later ones, cancels the activation, kills shell commands still running, removes the session directory, then disconnects from a lent sandbox or stops its own. Returns the observations. |

CrewAI 1.15 runs its agent loop as a Flow whose steps call the model and the tools synchronously from worker threads
of the event loop's default executor, even under `akickoff()`. Each of those calls is handed back to the event loop,
so model requests use Gym's aiohttp client and commands use the session's sandbox connection, and close can cancel
them. Those calls need the executor too, so the first seed sizes it to `concurrency` plus the default executor's
size, min(32, CPUs + 4): crews waiting on their calls cannot take every thread.

### Model calls

Every CrewAI model call, including the summarization CrewAI runs when a conversation nears the model's context
window, is one Chat Completions request to the Model Server's rollout-prefixed route
(`/ng-rollout/<capture key>[/training-token-capture]/v1/chat/completions`), with `x-session-id` set to the invocation
ID and the Model Server's cookies sent back. HTTP 408, 425, 429 and 5xx are retried `max_model_retries` times with
backoff; CrewAI also retries a rate-limited call up to three times on its own. CrewAI's own message keys, such as
`cache_breakpoint`, are dropped before the request. Reasoning that the model returns is kept in the transcript; CrewAI
does not send it back in later turns.

CrewAI's telemetry, trace upload, PyPI version check and replay database in the user's home are switched off when
the adapter imports CrewAI.

### Tools

| Tool | Source | Behavior |
| --- | --- | --- |
| `shell` | `SandboxAccess`, or the agent's own sandbox | Each call runs a fresh `bash` in the task directory with the owner's `SandboxAccess.env`, so `cd` and variables do not carry over. The command runs in a process group of its own and is killed with it after `shell_timeout_seconds`. Its combined stdout and stderr go to a file, so a background process it leaves cannot hold the exec open; background processes survive the command as task services. The runner is sent with each command, so a command that deletes the session directory does not break the next one. The model sees the output (head and tail beyond `shell_max_output_bytes`) and the exit code. |
| Each tool of an `mcp` grant | `tool_accesses` | Listed at seed with the grant's URL and headers; each call opens a fresh MCP session over Gym's aiohttp client, as CrewAI's own MCP tools open a fresh client per call. Tools keep their own names, which verifiers compare, rather than CrewAI's server-prefixed names. A server error result, or a JSON-RPC error such as invalid arguments, reaches the model as a CrewAI `ToolFailure`. |

The task image needs `bash`, `head`, `tail`, `wc`, `sleep`, `cat`, `rm` and `mkdir`; nothing is installed in the
sandbox.

Tool results reach the model whole, the shell's within `shell_max_output_bytes`: the adapter turns off CrewAI's
`use_better_compaction`, which would move a large result to a file on the Agent Server host and give the model an
extra tool to read it.

Seeds the adapter cannot serve answer 422: a required grant other than `mcp`, a sandbox connection other than
`direct`, a relative `workdir` (lent or in `sandbox_config`), a `SandboxAccess.user` override, and tool names that collide (with `shell` or with
each other) after CrewAI sanitizes them. A required MCP grant that does not list its tools fails the seed with 500,
which the Environment Server retries; an optional one is skipped with an `mcp_server_skipped` gap.

A sandbox or MCP grant that fails underneath the crew stops it at its next call, and the activation answers 503, so
the episode is retried instead of graded.

### Request fields

| Field | Handling |
| --- | --- |
| `input` | Text `user` messages become the CrewAI task's description, as is: the crew starts without inputs, so CrewAI does not read braces as template variables. Text `system` and `developer` messages, and `instructions`, join the agent's backstory. Other items are refused. |
| `tools` | Each function tool must name a tool of an MCP grant; the crew is offered those grant tools (and `shell`). Without `tools`, every grant tool is offered. |
| `temperature`, `top_p`, `max_output_tokens` | Sent on every model call; `max_output_tokens` caps each call, not the episode. |
| `parallel_tool_calls` | `false` is sent to the model with the tools. |
| `model` | Sent as the request's model name; the vLLM Model Server substitutes its own. |
| `metadata` | `agent_timeout_sec` lowers `timeout`, as the seed's `agent_timeout_sec` does. Other keys are ignored. |
| Others | Refused with 422 when set to a non-default value. |

### Results and evidence

The response holds the crew's transcript: reasoning items, assistant messages, function calls and the tool results
CrewAI sent to the model. Usage is the sum over every model call, or unknown with a `model_call_usage_unavailable`
gap when one call reported none. A run that hits `timeout` answers `incomplete` with its partial work, which is graded.
A final answer cut off at the output cap answers `incomplete` with `max_output_tokens`. A failed model request answers
`failed` with `ng_agent_outcome` naming `model_request_error` and its status; a transport failure names
`model_transport_error`; an answer with neither text nor a tool call, which CrewAI asks for once more before it raises
`EmptyLLMResponseError`, names `agent_invalid_output`; any other CrewAI error is `unknown`. CrewAI does not run a
failed task again (`max_retry_limit=0`): the Environment Server's outcome policy decides whether the episode is
verified or retried. When CrewAI reaches `max_iter` it asks the model for a final answer, and the run counts as
completed, as CrewAI counts it.

Close returns an `AgentObservationBundle` from `crewai_agent` with one `AgentInvocation` (the response's status,
`stop_reason` `timeout` when the run hit `timeout`, every model call's response ID and the conversation) and a
`ToolCallObservation` per tool call with executor timing and status: a nonzero exit is `failed`, a command's deadline
is `timeout`, and a call still running when the run's own deadline ended it is `cancelled`. CrewAI does not pass a
call's ID to the tool, so executions are joined to the model's calls by name and arguments, exact matches first; an
execution that cannot be joined is a `tool_call_identity_unavailable` gap. Cached and reasoning token counts that a
call does not report are gaps. A close before any activation returns an `observation_capture_failed` gap; a close that
interrupts a run adds `agent_activation_interrupted`.

The bundle also holds the agent's `SandboxObservation` of the sandbox it used. Its outcome is how the crew's run
ended: `completed`, `failed` with the invocation's error type, `timeout` at the run's deadline, or `cancelled` when
close came before the run finished. A failed provider teardown makes it `sandbox_error`. Close sends no trajectory of
its own: rollout collection builds the turns from the captured calls, and joins every call that carried the session
header to the invocation, including failed attempts that have no response ID.

## Configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `model_server` | `policy_model` | The Gym Model Server. |
| `model` | `${policy_model_name}` | Model name sent when the request names none. |
| `timeout` | 1800 | Seconds a crew run may take; a task's `agent_timeout_sec` lowers it. |
| `max_iter` | 50 | CrewAI `Agent.max_iter`. |
| `max_model_retries` | 2 | Retries of transient model HTTP statuses. |
| `context_window_size` | `null` (131072 in `configs/crewai_agent.yaml`) | The model's context window; CrewAI summarizes near 85% of it. Unset, CrewAI assumes 4,096 tokens. |
| `role`, `goal`, `backstory`, `expected_output` | Generic engineer prompts | The CrewAI agent and task text around the instruction. |
| `shell_timeout_seconds` | 600 | Deadline for one command. |
| `shell_max_output_bytes` | 30000 | Command output shown to the model. |
| `mcp_timeout_seconds` | 300 | Deadline for one MCP listing or call. |
| `concurrency` | 16 | Crews running at once; it also sizes the event loop's default executor. Keep it at least the Environment Server's `max_concurrent_episodes`: a seeded session that waits for a slot holds its sandbox outside `timeout`, and the `ttl_s` of an agent-owned sandbox does not cover the wait. |
| `sandbox_provider` | `null` | Provider block for the agent's own sandbox, used only when the Environment Server lends none. |
| `sandbox_config` | `{}` | `SandboxSpec` fields for that sandbox; `workdir` defaults to `/app`, `ttl_s` to `timeout` plus 600. |
| `session_close_timeout_seconds` | 60 | Budget for each close step. |

## Limitations

- A seed without `SandboxAccess` or `sandbox_provider` runs with grant tools only.
- No host path or `/run` route.
- `SandboxAccess.user` is refused: the shell's session files are written as the sandbox's default user.
- Direct HTTP tool grants are not consumed; a required one is refused and an optional one is skipped.
- Tool results that CrewAI never sent to the model (the run ended right after the call) are missing from the
  transcript. Tool-call records carry timing and status but no output, so the observations do not hold them either.
- A tool name that CrewAI's sanitizer changes (for example `getWeather` to `get_weather`) is offered to the model in
  the sanitized form; the transcript records the tool's own name.
- CrewAI creates an encryption key for its credential store under the Agent Server user's data directory
  (`~/.local/share/crewai/credentials`) on its first run; Gym never logs in to CrewAI.

## Tests

```bash
gym env test +entrypoint=responses_api_agents/crewai_agent
```

The tests run real crews against a scripted Model Server, real `bash` through the local sandbox provider, and a real
streamable-HTTP MCP server. The kill tests run on Linux only: macOS's bash 3.2 sometimes leaves a subshell of a killed
process group running.

The harness conformance probes give the agent its own local sandbox through `sandbox_provider` and run on Linux:

```bash
python scripts/run_harness_conformance.py --harness crewai --output <new directory>
```

## Licensing information

Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- crewai: MIT
