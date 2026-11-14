# CrewAI Agent

This adapter runs a [CrewAI](https://github.com/crewAIInc/crewAI) crew as a Gym harness, one crew turn per agent
session, inside the episode's task sandbox. The sandbox is the one the Resources Server lends the session, which the
adapter never stops, or, when it lends none and `sandbox_provider` is set, one the adapter starts and stops itself. At
seed the Agent Server installs a private Python runtime with CrewAI in the sandbox; the activation runs
[`sandbox_runner.py`](sandbox_runner.py) there under Gym's process supervisor. The crew's tools are a shell in the
task's working directory and the tools of the episode's MCP grants, and its model is the Gym Model Server.

- Harness: `crewai==1.15.34`, pinned in [`pyproject.toml`](pyproject.toml) and locked in [`uv.lock`](uv.lock). The
  sandbox runtime installs the same version, with its dependencies at this server's versions.
- Execution: a sandboxed harness under an Environment Server such as `single_agent_turn`, as
  [Hermes](../hermes_agent/README.md) is. There is no legacy episode execution or host path: `/run` uses Gym's standard
  terminal 422 refusal; `/v1/responses` without a seeded agent session answers 409,
  and a seed with no sandbox to run in answers 422.
- Capability level: Observability, plus Environment tools and task sandboxes. The close bundle carries an invocation
  record with every model-call reference, tool-call records with executor timing, and the agent's sandbox record. The
  [harness conformance probes](../../scripts/harness_conformance/README.md) for `crewai` meet P0 in all eight
  scenarios.
- Training: not qualified. `token_id_capture` is untested, and the runner's report is a file that the model's own
  commands can write.

## Configure a pairing

[`configs/crewai_agent.yaml`](configs/crewai_agent.yaml) declares the agent `crewai_agent` with
`resources_server: null`, so the Environment Server binds the benchmark. Keep `num_workers: 1`: sessions live in the
process that seeded them.

The crew calls the Model Server, and the MCP grants, from inside the sandbox, so both must be reachable from it at the
addresses Gym gives out. Set `use_absolute_ip: true` unless the sandbox shares the host's network: a seed whose model
URL is a loopback address is refused for a remote provider, and only logs a warning for local Docker.

### Terminal-Bench 2.1, in the task sandbox

Prepare the data as the [benchmark README](../../benchmarks/terminal_bench_2_1/README.md) describes, and save this as
`run.yaml`:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/crewai_agent/configs/crewai_agent.yaml
  - environment_servers/single_agent_turn/configs/single_agent_turn.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml
  - nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

use_absolute_ip: true
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
`OPENSANDBOX_API_KEY`. This is a development pairing: the benchmark does not declare `crewai_agent` in its
`allowed_agents`, and the full benchmark evaluation is pending. Pass `--allow-unsupported-pairing` to both commands.
Materialized task rows take no request overrides, so put a per-call output cap on the Model Server
(`sampling_overrides: {max_tokens: 8192}`) rather than on `gym eval run`:

```bash
gym env start --config run.yaml --allow-unsupported-pairing
gym eval run --no-serve --config run.yaml --allow-unsupported-pairing \
  --input results/tb21-input.jsonl --output results/crewai-tb21/rollouts.jsonl --concurrency 1
```

### A Resources Server's tools, as an MCP grant

Swapping the benchmark changes configuration only. This composition runs
[`example_single_tool_call`](../../resources_servers/example_single_tool_call/README.md) with its tools exposed over
MCP. That benchmark lends no sandbox, so the agent starts its own from `sandbox_provider`, here a Docker container:

```yaml
config_paths:
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml
  - responses_api_agents/crewai_agent/configs/crewai_agent.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml
  - nemo_gym/sandbox/providers/docker/configs/docker.yaml

use_absolute_ip: true

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

crewai_agent:
  responses_api_agents:
    crewai_agent:
      sandbox_provider: sandbox
      sandbox_config:
        image: python:3.13-slim
        workdir: /workspace
```

```bash
gym eval run --no-serve --config run.yaml --agent crewai_agent \
  --input resources_servers/example_single_tool_call/data/example.jsonl --output results/crewai-mcp/rollouts.jsonl
```

## How a session runs

| Call | What the adapter does |
| --- | --- |
| Seed (`/v1/agent_sessions`) | Connects to the lent sandbox, or starts its own from `sandbox_config`, then installs the CrewAI runtime unless one with the same key is already there, and stages the runner's files. A failed install releases a lent sandbox and stops the agent's own. |
| Activation (`/ng-rollout/<capture key>/v1/responses`) | Uploads the activation's input to `/tmp/nemo-gym-crewai-sessions/<id>`, outside the task directory, and runs the runner under Gym's process supervisor with the run's deadline. The runner builds one CrewAI agent and one task and runs `Crew.akickoff()` once. Identical retries join the same run; a different request answers 409. |
| Close (`/v1/agent_sessions/close`) | Stops the runner through the supervisor, which reports when the runner's process group is gone, reads the runner's report, removes the session directory, then disconnects from a lent sandbox or stops its own. Returns the observations. |

The supervisor sends the runner `SIGTERM` at the deadline, on close and on a rollout stop. The runner then cancels the
crew's calls in flight, kills a running shell command, and writes its report before it exits.

### Runtime

The runtime is a pinned standalone CPython with CrewAI, `aiohttp` and `mcp` installed at this server's versions,
built by the Agent Server for the sandbox's architecture and C library with
[`SandboxPythonRuntime`](../../nemo_gym/agents/utils/sandbox_python.py) and cached under `sandbox_python` in Gym's
`cache_dir`. Building needs `uv` on the Agent Server and a wheel of every dependency for the sandbox's platform. The
sandbox needs no Python, network or `uv` of its own. The runtime lives in `/tmp/nemo-gym-crewai-runtime-<key>`,
outside the task directory and `HOME`, and a later session in the same sandbox reuses it. Each new sandbox receives
it as two archives: for x86_64 glibc, about 35 MB of CPython and 275 MB of CrewAI and its dependencies, which include
LanceDB and ONNX Runtime. Building them the first time takes the Agent Server a minute or two.

The runner gives CrewAI a private `HOME` and data directory in the session directory, switches off its telemetry,
trace upload and PyPI version check, and starts shell commands with the task's own environment.

### Model calls

Every CrewAI model call, including the summarization CrewAI runs when a conversation nears the model's context
window, is one Chat Completions request from the sandbox to the Model Server's rollout-prefixed route
(`/ng-rollout/<capture key>[/training-token-capture]/v1/chat/completions`), with `x-session-id` set to the invocation
ID and the Model Server's cookies sent back. HTTP 408, 425, 429 and 5xx are retried `max_model_retries` times
(five retries by default). A positive finite `Retry-After` number or HTTP date sets the wait, capped at 60 seconds;
otherwise the wait uses full jitter with a one-second exponential base and the same 60-second cap. The default's
five waits total at most 31 seconds without `Retry-After`. Stop and the activation deadline cancel a pending wait.
Other 4xx responses are not retried, and exhaustion retains the final HTTP status. CrewAI also retries a rate-limited
call up to three times on its own. CrewAI's own message keys, such as
`cache_breakpoint`, are dropped before the request. Reasoning that the model returns is kept in the transcript; CrewAI
does not send it back in later turns.

CrewAI 1.15 runs its agent loop as a Flow whose steps call the model and the tools synchronously from worker threads,
even under `akickoff()`. The runner hands each of those calls to its event loop, so the stop can cancel them.

### Tools

| Tool | Source | Behavior |
| --- | --- | --- |
| `shell` | The task sandbox | Each call runs a fresh `bash` in the task directory with the task's environment and the owner's `SandboxAccess.env`, so `cd` and variables do not carry over. The command runs in a process group of its own and is killed with it after `shell_timeout_seconds`. Its combined stdout and stderr go to a file, so a background process it leaves cannot hold the call open; background processes survive the command as task services. The model sees the output (head and tail beyond `shell_max_output_bytes`) and the exit code. |
| Each tool of an `mcp` grant | `tool_accesses` | Listed when the activation starts, with the grant's URL and headers; each call opens a fresh MCP session from the sandbox, as CrewAI's own MCP tools open a fresh client per call. Tools keep their own names, which verifiers compare, rather than CrewAI's server-prefixed names. A server error result reaches the model as a CrewAI `ToolFailure`. |

The session token in each grant is readable inside the sandbox; it gives access only to that episode's tools.

Tool results reach the model whole, the shell's within `shell_max_output_bytes`: the runner turns off CrewAI's
`use_better_compaction`, which would move a large result to a temporary file and give the model an extra tool to read
it.

Seeds the adapter cannot serve answer 422: a required grant other than `mcp`, no lent sandbox and no
`sandbox_provider`, a sandbox connection other than `direct`, a relative `workdir` (lent or in `sandbox_config`), a
`SandboxAccess.user` override, and a loopback model URL for a remote sandbox provider.
Activations that ask for tools the grants cannot provide answer 422: tool names that collide (with `shell` or with each
other) after CrewAI sanitizes them, and request tools that no grant serves. A required MCP grant that does not list its
tools fails the activation with 503 before the first model call, which the Environment Server retries; an optional one
is skipped with an `mcp_server_skipped` gap.

An MCP grant or the sandbox's `bash` that fails underneath the crew stops it at its next call, and the activation
answers 503, so the episode is retried instead of graded. Its failure header records a stable
`harness/agent_run_error` event with the operation (`shell_start`, `mcp_list_tools` or `mcp_call_tool`), grant name when
applicable, and bounded exception cause facts. An observed MCP connection failure or timeout retains its transport
kind; an HTTP rejection retains its actual status. These are runner failures, not invented sandbox-provider errors;
actual provider failures retain their shared-layer classifications. Close replays the same event identities.
Diagnostics select exception type, OS error text or HTTP status without serializing request URLs, headers or bodies.

### Request fields

| Field | Handling |
| --- | --- |
| `input` | Text `user` messages become the CrewAI task's description, as is: the crew starts without inputs, so CrewAI does not read braces as template variables. Text `system` and `developer` messages, and `instructions`, join the agent's backstory. Other items are refused. |
| `tools` | Each function tool must name a tool of an MCP grant; the crew is offered those grant tools (and `shell`). Without `tools`, every grant tool is offered. |
| `temperature`, `top_p` | Sent on every model call. |
| `max_output_tokens` | Non-null values are refused with 422 before the activation is bound; the crew cannot enforce an aggregate response budget. Set per-call caps on the Model Server instead. A corrected request can use the same seeded session. |
| `parallel_tool_calls` | `false` is sent to the model with the tools. |
| `model` | Sent as the request's model name; the vLLM Model Server substitutes its own. |
| `metadata` | `agent_timeout_sec` lowers `timeout`, as the seed's `agent_timeout_sec` does. Other keys are ignored. |
| Others | Refused with 422 when set to a non-default value. |

### Results and evidence

The response holds the crew's transcript: reasoning items, assistant messages, function calls and the tool results
CrewAI sent to the model. Usage is the sum over every model call, or unknown with a `model_call_usage_unavailable`
gap when one call reported none. A run that hits `timeout` answers `incomplete` with its partial work, which is graded;
a rollout stop does the same and adds its `ng_rollout_stop` marker. A final answer cut off at the output cap answers
`incomplete` with `max_output_tokens`. A failed model request answers `failed` with `ng_agent_outcome` naming
`model_request_error` and its status; a transport failure names `model_transport_error`; an answer with neither text
nor a tool call, which CrewAI asks for once more before it raises `EmptyLLMResponseError`, names
`agent_invalid_output`. An unclassified CrewAI exception returns `failed` with
`ng_agent_outcome.failure_kind="unknown"`; the default Environment Server outcome policy retries that episode.
CrewAI does not run a failed task again
(`max_retry_limit=0`): the Environment Server's outcome policy decides whether the episode is verified or retried.
When CrewAI reaches `max_iter` it asks the model for a final answer, and the run counts as completed, as CrewAI counts
it.

Close returns an `AgentObservationBundle` from `crewai_agent` with one `AgentInvocation` (the response's status,
`stop_reason` `timeout` when the run hit `timeout`, every model call's response ID and the conversation) and a
`ToolCallObservation` per tool call with executor timing and status: a nonzero exit is `failed`, a command's deadline
is `timeout`, and a call still running when the run's own deadline ended it is `cancelled`. CrewAI does not pass a
call's ID to the tool, so executions are joined to the model's calls by name and arguments, exact matches first; an
execution that cannot be joined is a `tool_call_identity_unavailable` gap. Cached and reasoning token counts that a call does not report are gaps. A close
before any activation returns an `observation_capture_failed` gap; a close that interrupts a run adds
`agent_activation_interrupted`, and a run that hits its deadline adds `agent_run_timeout`.

The bundle also holds the agent's `SandboxObservation` of the sandbox the crew ran in. Its outcome is how the crew's
run ended: `completed`, `failed` with the invocation's error type, `timeout` at the run's deadline, `cancelled` when
close came before the run finished, or `oom` when the provider confirms the runner was killed for memory. A sandbox
the agent started and stopped has `teardown_confirmed`. Close reports `cleanup_confirmed` from the supervisor's
receipt and sends no trajectory of its own: rollout collection builds the turns from the captured calls, and joins
every call that carried the session header to the invocation, including failed attempts that have no response ID.

## Configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `model_server` | `policy_model` | The Gym Model Server. |
| `model` | `${policy_model_name}` | Model name sent when the request names none. |
| `timeout` | 1800 | Seconds a crew run may take; a task's `agent_timeout_sec` lowers it. |
| `max_iter` | 50 | CrewAI `Agent.max_iter`. |
| `max_model_retries` | 5 | Additional attempts for transient model HTTP statuses; 0 disables these retries. Backoff/`Retry-After` waits are capped at 60 seconds each and remain inside the activation deadline. |
| `context_window_size` | `null` (131072 in `configs/crewai_agent.yaml`) | The model's context window; CrewAI summarizes near 85% of it. Unset, CrewAI assumes 4,096 tokens. |
| `role`, `goal`, `backstory`, `expected_output` | Generic engineer prompts | The CrewAI agent and task text around the instruction. |
| `shell_timeout_seconds` | 600 | Deadline for one command. |
| `shell_max_output_bytes` | 30000 | Command output shown to the model. |
| `mcp_timeout_seconds` | 300 | Deadline for one MCP listing or call. |
| `concurrency` | 16 | Crews running at once. Keep it at least the Environment Server's `max_concurrent_episodes`: a seeded session that waits for a slot holds its sandbox outside `timeout`, and the `ttl_s` of an agent-owned sandbox does not cover the wait. |
| `sandbox_provider` | `null` | Provider block for the agent's own sandbox, used only when the Environment Server lends none. |
| `sandbox_config` | `{}` | `SandboxSpec` fields for that sandbox; `workdir` defaults to `/app`, `ttl_s` to `sandbox_install_timeout_seconds` plus `timeout` plus 600. |
| `sandbox_install_timeout_seconds` | 900 | Time limit for installing the runtime at seed. |
| `network_access` | `inherit` | Egress of the agent's own sandbox: `inherit` (the provider's network), `model_only` (the model server's host) or `model_and_tools` (also each MCP grant's host), applied through `restrict_sandbox_network`. A seed that lends a sandbox refuses any other value than `inherit`, and so does a provider without network policies. The runtime is uploaded from the Agent Server, so a restricted sandbox still gets it. |
| `session_close_timeout_seconds` | 60 | Budget for each close step. |

## Limitations

- No host execution or legacy `/run` execution; `/run` uses the framework's terminal 422 refusal. No session without a sandbox.
- `SandboxAccess.user` is refused: the runtime and the session's files are written as the sandbox's default user.
- Direct HTTP tool grants are not consumed; a required one is refused and an optional one is skipped.
- The runner and the model's shell commands run as the same user in the same sandbox. A command such as
  `pkill python3` can stop the runner, and a command can write the runner's report.
- Tool results that CrewAI never sent to the model (the run ended right after the call) are missing from the
  transcript. Tool-call records carry timing and status but no output, so the observations do not hold them either.
- A tool name that CrewAI's sanitizer changes (for example `getWeather` to `get_weather`) is offered to the model in
  the sanitized form; the transcript records the tool's own name.

## Tests

```bash
gym env test +entrypoint=responses_api_agents/crewai_agent
```

The tests run real crews in the sandbox runner, with the host's CrewAI standing in for the sandbox runtime, against a
stub Model Server, real `bash` through the local sandbox provider, and a real streamable-HTTP MCP server. The episode
tests run on Linux only, because Gym's process supervisor needs it; so do the kill tests, because macOS's bash 3.2
sometimes leaves a subshell of a killed process group running.

The harness conformance probes give the agent its own local sandbox through `sandbox_provider` and run on Linux:

```bash
python scripts/run_harness_conformance.py --harness crewai --output <new directory>
```

## Licensing information

Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- crewai: MIT
