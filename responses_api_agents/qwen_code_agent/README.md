# Qwen Code Agent

Runs the [Qwen Code](https://github.com/QwenLM/qwen-code) CLI **0.25.1** (Apache-2.0) headless inside a Gym task
sandbox. The Environment Server seeds the Resources and agent sessions, activates Qwen Code once, closes the agent,
verifies the task, and closes Resources. Qwen Code runs its own tools and calls Gym Model Servers for inference.
Its native subagents (the `agent` tool, nested delegation, background agents and the experimental Agent Team) are
configurable.

Capability level: **Evaluation**. The adapter returns the root session as Responses items, Qwen Code's run usage
(subagents included), and an observation bundle with explicit gaps.

Model calls already use the rollout's capture route, but no training capture run has validated this harness yet.
Capture is refused at seed while `subagent_model_server` is set, since its calls would land in the policy rollout.
The harness conformance probes are not run at the Evaluation level.

## Configure and run

[configs/qwen_code_agent.yaml](configs/qwen_code_agent.yaml) defines the agent defaults. For SWE-bench Pro, save this
composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/qwen_code_agent/configs/qwen_code_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: qwen_code_agent
      resources_tool_transports: []
```

Supply a `policy_model` Gym Model Server, `policy_model_name`, and the `sandbox` provider configuration in
`model-provider.yaml`. The task sandbox must be able to reach the Model Server.
The SWE-bench Pro pairing is undeclared (this agent is not in that Resources server's `allowed_agents`), so
`gym env start` needs `--allow-unsupported-pairing`.

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml --allow-unsupported-pairing

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent qwen_code_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

## Solo and swarm settings

| Field | Default | Effect |
|---|---|---|
| `subagents` | `true` | `false` removes the `agent` and `list_agents` tools: the solo control. |
| `subagent_model` | `null` | Route id for the built-in `general-purpose` subagent. `null` inherits the root model. |
| `subagent_model_server` | `null` | Gym Model Server behind `subagent_model`, so children run a different model. Required with `subagent_model`. |
| `max_subagent_depth` | `null` (Qwen Code: 5) | Nesting levels, 1 to 100. 1 keeps subagents but forbids nesting. |
| `max_parallel_agents` | `null` (Qwen Code: 10) | Concurrent background subagents. |
| `agent_team` | `false` | Enables the experimental Agent Team tools. |
| `max_session_turns`, `max_tool_calls` | `null` | `--max-session-turns` and `--max-tool-calls`. A request's `max_tool_calls` wins. Qwen Code counts tool calls in the root session only, so subagent calls are not limited. |
| `approval_mode` | `yolo` | Headless runs need `yolo` (or `auto-edit`, `auto`) to execute tools. |
| `timeout` | `3600` | Seconds per activation, children included, enforced by the sandbox's process supervisor. |
| `exclude_tools` | `[web_fetch, web_search]` | Extra `--exclude-tools`. The web tools are off by default, so tasks are solved without looking answers up online. Set `[]` to turn web access on. A configured list replaces the default. |
| `settings` | `{}` | Merged under the generated settings (the model routes it sets win). Its `model.generationConfig` (for example `contextWindowSize`, `samplingParams.max_tokens`, `extra_body`) is copied into every model route, because a route's own `generationConfig` wins per field over the top-level block. |
| `network_access` | `inherit` | Egress of an agent-owned sandbox. `model_only` allows only the model servers' host and requires `runtime_archive`. |

`subagent_model` works by writing a user-level `general-purpose` agent definition with that `model`. Its system
prompt is the built-in one, copied from Qwen Code 0.25.1 into [general-purpose-agent.md](general-purpose-agent.md).
Forks and teammates keep the root model.

## Runtime and sandbox setup

Session setup installs a private Node.js 22.23.2 and `@qwen-code/qwen-code@0.25.1` (npm, `--ignore-scripts`) under
`/tmp/nemo-gym-qwen-runtime-0.25.1-node-22.23.2`, checks the version Qwen Code reports, and reuses a runtime that
already reports it. The agent server downloads the official Node.js build for the sandbox's architecture once into
its asset cache, checks the SHA-256 pinned in `sandbox.py`, and uploads it; the sandbox checks it again. Qwen Code runs
as that Node with the package's entry script, so the task's own Node, npm and `PATH` are untouched, and the task's
`NODE_OPTIONS` and `NODE_PATH` are unset. A sandbox that already has a ready runtime skips the upload and install. The
sandbox needs glibc 2.28 or newer and `flock` (other images are refused at seed), bash, `sha256sum`, `tar` and network
access to the npm registry. For sandboxes without internet access, build a runtime archive once with
[build_runtime.sh](build_runtime.sh) for the sandbox's architecture and set `runtime_archive`. The agent server
uploads it as is, and the installer checks the version it reports:

```bash
responses_api_agents/qwen_code_agent/build_runtime.sh /path/to/qwen-runtime-x64.tar.gz 0.25.1 x64
```

Qwen Code runs in `SandboxAccess.workdir` through the shared `harness_runner.sh` under the process supervisor. Its
settings, agent definitions and session transcripts live in a private session directory, so tool shells keep the
task's `HOME`. The settings load at Qwen Code's system scope (`QWEN_CODE_SYSTEM_SETTINGS_PATH`), which outranks a task
repository's `.qwen/settings.json`. The models.dev catalog refresh is off. `SandboxAccess.env` is forwarded, except `QWEN_*`, `OPENAI_*` and the adapter's
`NEMO_GYM_MODEL_API_KEY`. Close removes only the session directory.

`network_access: model_only` limits an agent-owned sandbox's egress to the model servers' host (every port). npm cannot
reach its registry then, so the configuration requires `runtime_archive`. The
OpenSandbox policy is set when the sandbox is created, so a session that borrows a task sandbox rejects any value
other than `inherit`, as does a provider without network policies, an explicit
`sandbox_config.provider_options.network_policy`, or a `subagent_model_server` on another host. Qwen Code uses no tool
grants, so `model_and_tools` is rejected. Each refusal fails the seed with HTTP 422.

## Requests, results and outcomes

The user message is the task, passed on stdin. The request's system or developer message, `instructions` and
`system_prompt` form `--append-system-prompt` (at most 128 KiB). Request `temperature` and `top_p` go into every model
route's `generationConfig`, so they apply to every model call. `max_output_tokens` is refused, because Qwen Code cannot enforce a total budget. Other request fields are
refused with HTTP 422, as are sessionless requests and `/run`.

The response output is the root session from `-o stream-json`: reasoning, messages, tool calls and results. Qwen Code
defers MCP tools and the `agent` tool behind its `tool_call` bridge; a `tool_call` is recorded as the underlying call.
Usage is the result event's run total, which includes subagent calls. Response metadata records `qwen_exit_code`,
`qwen_subagents`, `qwen_subagents_background`, `qwen_subagent_max_depth`, `qwen_subagent_input_tokens` and
`qwen_subagent_output_tokens`, from the exported session transcripts.

| Qwen Code 0.25.1 ending | Response |
|---|---|
| Exit 0, a successful result event and an assistant message | `completed` |
| Supervisor deadline or rollout stop, with or without model output | `incomplete` (verified) |
| Exit 53 (`--max-session-turns`), exit 55 (`--max-tool-calls`) or loop detection | `incomplete` (verified) |
| `is_error` result with `[API Error: <status> ...]` | `failed`, `model_request_error` with that `status_code` |
| Any other nonzero exit (a bare 137 included), no result event, or `is_error` result | `failed`, `unknown` |
| Exit 0 without an assistant message | `failed`, `agent_invalid_output` |
| Qwen Code never started | The activation raises |

A subagent error the root recovers from does not fail the run: only the root session's result event and exit code
count. A tool result with `is_error` is recorded with status `incomplete`.

## Limitations

- Model-call ownership, tool timing and child invocations are not reported (`model_call_ownership_unavailable`,
  `harness_tool_timing_unavailable`, `child_invocations_unavailable`).
- Required tool grants are refused at seed; optional grants are skipped and reported. A `SandboxAccess` user override
  is refused.
- Agent Team teammates do not write session files; their prompts appear only in stdout.
- Cron tools are off (`experimental.cron: false`), because a headless run waits for its scheduled jobs.
- One activation per session; there is no host execution path.

## Tests

```bash
python -m pytest responses_api_agents/qwen_code_agent/tests -q -o addopts=''
```

The lifecycle tests run the shared process supervisor and runner on Linux with a fake Qwen Code CLI.
