# Terminus-2 Agent

Runs [Harbor's Terminus-2](https://github.com/harbor-framework/harbor/tree/main/src/harbor/agents/terminus_2)
as a NeMo Gym agent. Terminus-2 drives one tmux terminal: it types the keystrokes it reads from the model's
JSON (or XML) reply and reads the terminal back. It uses no function calling.

- Harness: Harbor at commit `9dddd797b57ab8a0f9d6352a20fce73abbb29573`, pinned in `pyproject.toml`.
- Model API: Chat Completions, through the Gym Model Server named by `model_server`.
- Capability level: Observability. Agent sessions report the invocation, its model calls by response ID, each
  command batch with the terminal output Terminus recorded for it, summarizations, and the sandbox outcome.
- Tool grants from Resources: none. A seed with a required grant is refused; keep `resources_tool_transports: []`.

## Execution modes

| Request | Where Terminus-2 runs | Cleanup |
| --- | --- | --- |
| Agent session seeded with `sandbox_access` | Inside the task sandbox the Resources Server lends. | Close stops Terminus and removes its session files. Resources destroys the sandbox. |
| Agent session without `sandbox_access`, with `sandbox_provider` set | Inside a sandbox the agent starts from `sandbox_provider` and `sandbox_config`. | Close stops the sandbox. |
| `/v1/responses` without a session, and `/run` | On the Agent Server host, in a temporary workspace (or `workspace_root`). | The agent removes its workspace and logs. |

Agent sessions run with `num_workers: 1`, the default. A sandbox failure never falls back to the host.

### Agent sessions

At seed, the agent prepares the sandbox before any activation:

1. It reuses or installs a private Python runtime with this server's Harbor, built on the Agent Server with
   `SandboxPythonRuntime` and uploaded, so the sandbox needs no Python and no network for it. Packages are pinned
   to the versions in this server's environment. Harbor's sandbox backends, dataset tools and CLI are left out,
   because Terminus-2 never imports them. A runtime an earlier session left in the sandbox is reused when it
   passes its import check. The runtime lives under `/tmp/nemo-gym-terminus-2-runtime-<key>`.
2. It uploads `sandbox_runner.py`, `terminus.py` and `llm.py` into the session directory,
   `/tmp/nemo-gym-terminus-2-sessions/<id>`, outside the task's working directory and `HOME`.
3. It checks that `script` (util-linux) and `tmux` run. A sandbox without `script` is refused. A sandbox without
   `tmux` gets it from Harbor's own installer, which uses the image's package manager, so such a sandbox needs
   network access to its package index. Install `tmux` in the image to avoid this.

The activation runs the runner under Gym's process supervisor in the task's working directory. The runner calls
the Model Server directly with the rollout-qualified URL and an `x-session-id` header that names the invocation,
so the sandbox must reach the Model Server (`++use_absolute_ip=true` for Docker). Terminus's terminal starts with
the environment the runner started with, so tool commands keep the task's `HOME`, `PATH` and Python settings.

When Terminus confirms completion, the runner exits and leaves the tmux terminal running, as Harbor does:
services the agent started in it stay up for verification, until the sandbox's owner stops the sandbox. A run
still going at the time limit, on a rollout stop, or on close is stopped by the supervisor with every process it
started, including the terminal. The runner saves the turns taken so far before it exits.

The response status says how the run ended:

| Ending | Status | Metadata `stop_reason` |
| --- | --- | --- |
| Terminus confirmed completion | `completed` | `completed` |
| `max_turns` reached, time limit, rollout stop, the terminal exited | `incomplete` | `max_turns`, `wall_time`, `wall_time`, `terminal_exited` |
| Context overflow that summarization could not resolve | `incomplete` | `context_limit` |
| Model request rejected or failed, other harness error | `failed`, with `ng_agent_outcome` | `error` |

A failed response names `model_request_error` with the model's HTTP status, `model_transport_error`, or
`unknown`, and the Environment Server's outcome policy decides whether to verify, retry or fail the episode.
Usage is reported only when every model response carried it.

## Configuration

`configs/terminus_2_agent.yaml` is the agent-session configuration (`resources_server: null`). Besides the Terminus
settings (`max_turns`, `parser_name`, `temperature`, `enable_summarize`, `model_info`, tmux pane size and others),
it sets:

| Field | Purpose |
| --- | --- |
| `timeout` | The run's wall-clock budget. A task's `agent_timeout_sec` lowers it in agent sessions. |
| `concurrency` | Concurrent runs, agent session activations or host runs. |
| `sandbox_provider`, `sandbox_config` | An agent-owned sandbox for sessions without `sandbox_access`; `sandbox_config.workdir` must be absolute. |
| `sandbox_install_timeout_seconds` | Runtime and tmux setup at seed. |
| `session_close_timeout_seconds` | Each step of close: stopping Terminus, collecting its result, releasing the sandbox. |

Request sampling fields are converted with Gym's vLLM converter and sent on every model call. `model_timeout_sec`
bounds each model call and `command_timeout_sec` each terminal command. `record_terminal_session` applies only on
the host path. `collect_rollout_details` returns token IDs and log probabilities; in a sandbox they come back in the
runner's result file, which the task's commands could also write, so treat them as evaluation evidence only.

### Run on Terminal-Bench 2.1

Terminal-Bench 2.1 does not list `terminus_2_agent` in `allowed_agents` yet, so start its servers with
`--allow-unsupported-pairing`. Compose the run as the
[benchmark README](../../benchmarks/terminal_bench_2_1/README.md#configure-a-run) shows, with
`responses_api_agents/terminus_2_agent/configs/terminus_2_agent.yaml` and `agent_server.name: terminus_2_agent`.

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
- Terminus-2 records no start or end time for a command batch, so tool timing is reported as unavailable.
- The tmux server uses the default socket under `/tmp/tmux-<uid>`; a task that runs its own tmux server shares it.
