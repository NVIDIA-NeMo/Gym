# Terminus-2 Agent

Runs [Harbor's Terminus-2](https://github.com/harbor-framework/harbor/tree/main/src/harbor/agents/terminus_2)
as a NeMo Gym agent. Terminus-2 drives one tmux terminal: it types the keystrokes it reads from the model's
JSON (or XML) reply and reads the terminal back. It uses no function calling.

- Harness: Harbor at commit `9dddd797b57ab8a0f9d6352a20fce73abbb29573`, pinned in `pyproject.toml`.
- Model API: Chat Completions, through the Gym Model Server named by `model_server`.
- Capability level: Evaluation. Agent sessions report the invocation, its model calls by response ID, each
  command batch with the terminal output Terminus recorded for it, summarizations, and the sandbox outcome, but the
  evidence check's TE-5 (`tools.output`) fails on turns with several commands until
  [#3520](https://github.com/NVIDIA-NeMo/Gym/pull/3520) lands. The level becomes Observability once it does and the
  evidence is rerun.
- Tool grants from Resources: none. A seed with a required grant is refused; keep `resources_tool_transports: []`.
- Supported pairings: none declared. Smoke-tested: Terminal-Bench 2.1 with a borrowed sandbox (unsupported-pairing
  override), and `math_with_judge` through `single_agent_turn_legacy` with an agent-owned sandbox. Terminal-Bench
  2.1 support awaits a full evaluation.

## Execution modes

| Request | Where Terminus-2 runs | Cleanup |
| --- | --- | --- |
| Agent session seeded with `sandbox_access` | Inside the task sandbox the Resources Server lends. | Close stops Terminus and removes its session files. Resources destroys the sandbox. |
| Agent session without `sandbox_access`, with `sandbox_provider` set | Inside a sandbox the agent starts from `sandbox_provider` and `sandbox_config`. | Close stops the sandbox. |
| `/v1/responses` without a session, and `/run` | On the Agent Server host, in a temporary workspace (or `workspace_root`). | The agent removes its workspace and logs. |

Agent sessions keep each session's state in the worker that seeded it, and Gym routes the session's later requests
to that worker; `concurrency` applies per worker. A sandbox failure never falls back to the host.

### Agent sessions

At seed, the agent prepares the sandbox before any activation:

1. It reuses or installs a private Python runtime with this server's Harbor, built on the Agent Server with
   `SandboxPythonRuntime` and uploaded, so the sandbox needs no Python and no network for it. Packages are pinned
   to the versions in this server's environment. Harbor's sandbox backends, dataset tools and CLI are left out,
   because Terminus-2 never imports them. A runtime an earlier session left in the sandbox is reused when it
   passes its import check. The runtime lives under `/tmp/nemo-gym-terminus-2-runtime-<key>`.
2. It uploads `sandbox_runner.py`, `terminus.py` and `llm.py` into the session directory,
   `/tmp/nemo-gym-terminus-2-sessions/<id>`, outside the task's working directory and `HOME`.
3. It checks that `bash`, `script` (util-linux) and `tmux` run. A sandbox without `bash` or `script` is refused.
   A sandbox without `tmux` gets it from Harbor's own installer, which uses the image's package manager, so such a
   sandbox needs root and network access to its package index. Install `tmux` in the image to avoid this.

In a sandbox it starts itself, the agent also creates `sandbox_config.workdir`, since providers need not.

The activation runs the runner under Gym's process supervisor in the task's working directory. The runner calls
the Model Server directly with the rollout-qualified URL and an `x-session-id` header that names the invocation,
so the sandbox must reach the Model Server. For Docker, set
`++use_absolute_ip=true '++absolute_ip_server_types=[responses_api_models]'`: only the Model Server needs the
address sandboxes reach, and an unseeded request to this agent runs Terminus on the Gym host. Terminus's terminal
starts with the environment the runner started with, so tool commands keep the task's `HOME` and Python settings.
The terminal runs `bash --login`, as Harbor's does, so the image's `/etc/profile` may reset `PATH`.

The tmux terminal and the commands running in it stay up after the runner ends, also after a deadline, rollout stop
or close, until the sandbox's owner stops the sandbox, as in Harbor: services the agent started in it stay up for
verification. A run still going at the time limit, on a rollout stop, or on close is stopped by the supervisor,
and the runner saves the turns taken so far before it exits.

The response status says how the run ended:

| Ending | Status | Metadata `stop_reason` |
| --- | --- | --- |
| Terminus confirmed completion | `completed` | `completed` |
| `max_turns` reached, time limit, rollout stop, the terminal exited | `incomplete` | `max_turns`, `wall_time`, `wall_time`, `terminal_exited` |
| Close during the run | `incomplete` | `cancelled` |
| A task command, such as `pkill python3`, sent the runner SIGTERM | `incomplete` | `terminated` |
| Context overflow that summarization could not resolve | `incomplete` | `context_limit` |
| Model request rejected or failed, other harness error | `failed`, with `ng_agent_outcome` | `error` |
| A stop reason this adapter does not know | `failed`, with `ng_agent_outcome` | the reason the runner saved |

A rollout stop or the time limit that arrives before the runner saved anything, during seed or while the runner
starts, ends the run as an empty `wall_time` run. Only the supervisor's receipt makes a SIGTERM a time limit. A failed response names `model_request_error` with the model's HTTP status,
`model_transport_error`, `agent_invalid_output` (a reply cut off at the output limit), or `unknown`, and the
Environment Server's outcome policy decides whether to verify, retry or fail the episode. In both execution modes,
usage is reported only when every model response carried it, and a model call cancelled by the time limit leaves it
unknown. Reasoning-token counts are always unknown, because Harbor keeps none.

## Configuration

`configs/terminus_2_agent.yaml` is the agent-session configuration (`resources_server: null`). Composing it onto a
benchmark does not replace the benchmark's own agent. For a benchmark whose Resources server lends no sandbox, use
`configs/profiles/benchmark.yaml` (below); for the host-path `/run`, set `resources_server` and front the agent with
the [`legacy_agent`](../../environment_servers/legacy_agent/README.md) Environment Server. Besides the Terminus
settings (`max_turns`, `parser_name`, `temperature`, `enable_summarize`, `model_info`, tmux pane size and others),
it sets:

| Field | Purpose |
| --- | --- |
| `timeout` | The run's wall-clock budget. A task's `agent_timeout_sec` lowers it in agent sessions. |
| `concurrency` | Concurrent runs, agent session activations or host runs. |
| `sandbox_provider`, `sandbox_config` | An agent-owned sandbox for sessions without `sandbox_access`; `sandbox_config.workdir` must be absolute. |
| `sandbox_install_timeout_seconds` | Each setup step at seed (runtime install, tmux install). An owned sandbox's default `ttl_s` reserves both, the run's `timeout`, and ten minutes for cleanup. |
| `session_close_timeout_seconds` | Each step of close: stopping Terminus, collecting its result, releasing the sandbox. |

Agent sessions treat the request's fields as follows:

- Applied: `instructions` and the input's system and developer messages, which join the instruction before the
  task; `temperature`, `top_p` and reasoning effort. Gym's
  Chat Completions converter converts them and the runner sends them on every model call.
- Ignored by design: `tools`, `tool_choice` and `parallel_tool_calls`, because Terminus types keystrokes and calls no
  tools. The input is flattened to text, so images and function-call history are not used.
- Refused with 422 before any sandbox work: fields the converter cannot represent, such as `previous_response_id`,
  `conversation`, `prompt`, `max_tool_calls`, `truncation` or `background`.

Request `max_output_tokens` depends on the execution mode:

- Host path (`/v1/responses` without an agent session, and `/run`): it becomes `max_tokens` on every model call, a
  per-call cap as before. `anyterminal_terminus_2` rows set it this way.
- Agent sessions: refused with HTTP 422 before the activation binds or the sandbox runs. A session's request
  describes the whole episode, and Terminus cannot enforce a budget shared across model calls. Remove
  `responses_create_params.max_output_tokens` from session task rows. To keep a per-call cap, set
  `policy_model.responses_api_models.vllm_model.extra_body.max_tokens`, for example `8192`, on the Chat Completions
  Model Server.

The agent's `model_info.max_output_tokens` describes the model's output limit to Terminus in both modes; it is not a
generation budget. Other sampling controls and model-server limits are unchanged.

`model_timeout_sec` bounds each model call and `command_timeout_sec` each terminal command. `reasoning_effort` and
`max_thinking_tokens` must stay `null`: Harbor applies them only to the LiteLLM client it builds itself, which this
agent replaces with Gym's Model Server client, so the agent refuses to start with either set. Set the request's
`reasoning` instead.
`record_terminal_session` applies only on the host path. `collect_rollout_details` returns token IDs and log
probabilities; in a sandbox they come back in the runner's result file, which the task's commands could also write,
so treat them as evaluation evidence only.

### Benchmark profile

`configs/profiles/benchmark.yaml` is a swappable profile for benchmarks whose Resources server supplies no sandbox,
such as HLE. It leaves `resources_server.name` unset (`???`), so Gym composes this agent onto the benchmark's own
agent instance, and each session creates its own sandbox through the `sandbox` provider. That sandbox must reach the
model server by address, so the profile sets `use_absolute_ip: true` and keeps the head server and the agent itself
on `127.0.0.1`: the head server's config endpoint is unauthenticated, and an unseeded request to the agent runs
Terminus on the Gym host. Terminus derives the sandbox's lifetime from its own timeouts, and its terminal starts in
`sandbox_config.workdir` (`/workspace`), which setup creates. Supply the image in `sandbox_config.image`; set other
`sandbox_config` keys under the profile's own instance name,
`terminus_2_agent.responses_api_agents.terminus_2_agent.sandbox_config`.

HLE, HLE Diamond, GPQA and APEX Shortlist need no Environment Server setting: their judges serve no tools, so their
Environment Servers grant the agent none. This agent rejects a required `direct_http` grant with a 422.

### Smoke-test on Terminal-Bench 2.1

Terminal-Bench 2.1 does not list `terminus_2_agent` in `allowed_agents`, so start its servers with
`--allow-unsupported-pairing`. Compose the run as the
[benchmark README](../../benchmarks/terminal_bench_2_1/README.md#configure-a-run) shows, with
`responses_api_agents/terminus_2_agent/configs/terminus_2_agent.yaml` and `agent_server.name: terminus_2_agent`.

### Relation to terminus_2_sandboxed_agent

[`terminus_2_sandboxed_agent`](../terminus_2_sandboxed_agent/README.md) also runs Harbor's Terminus 2 through agent
sessions, but its loop runs on the Agent Server and sends each command to the sandbox, with Harbor 0.22.0. This
adapter runs the whole loop inside the sandbox, with Harbor at the commit above. Terminal-Bench 2.1 declares only
`terminus_2_sandboxed_agent`. The two differ in Harbor version and in where commands run, so their scores are not
interchangeable.

## Host path and AnyTerminal

Without an agent session, Terminus-2 runs on this host in a fresh temporary workspace, or in `workspace_root`.
The host must provide `bash`, `tmux`, and `script`. `/run` uses this path; it requires `resources_server`.

Prepare the Terminal Bench dataset first:

```bash
python responses_api_agents/anyterminal_agent/prepare.py --no-build-sif
```

```bash
gym eval run \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_terminus_2.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --agent anyterminal_terminus_2 \
  --input responses_api_agents/anyterminal_agent/data/terminal_bench.jsonl \
  --output terminus_2_rollouts.jsonl
```

AnyTerminal runs this agent's host path inside each task container and sets `workspace_root: "."`.

The agent converts the Terminus-2 trajectory to Responses API messages and presents each command batch as a
`bash_command` function call with the terminal output as its result.

## Limits

- Harness conformance probes do not drive Terminus-2 yet: they expect a function-calling shell tool.
- The evidence check's TE-5 (`tools.output`) fails on turns with several commands until
  [#3520](https://github.com/NVIDIA-NeMo/Gym/pull/3520) lands, so the capability level is Evaluation.
- Terminus-2 records no start or end time for a command batch, so tool timing is reported as unavailable.
- The tmux server uses the default socket under `/tmp/tmux-<uid>`; a task that runs its own tmux server shares it.
- Sandbox runs are tested on glibc images; musl images are untested. Installing `tmux` at setup needs root and
  network access.
- A task command such as `pkill python` or `pkill -f '\.py'` can end the runner. SIGTERM ends the run as
  `terminated`, graded with the turns taken so far; SIGKILL leaves no result, and the activation fails as an
  infrastructure error.
- A completion Terminus confirms while the runner is closing its model client, at the same moment as the time limit,
  can be saved as `wall_time`.
- A model call still in flight at the deadline ends the run as `wall_time`, and the run is graded.
- The runner's result file (status, stop reason, token IDs) is writable by task commands, so treat it as
  evaluation evidence only.
