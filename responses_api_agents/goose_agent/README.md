# Goose Agent

Runs [Goose](https://github.com/aaif-goose/goose) **1.54.0** (Apache-2.0) headless with `goose run` inside a Gym
task sandbox. The Environment Server seeds the Resources and agent sessions, activates Goose once, closes the agent,
verifies the task, and closes Resources. Goose runs its own tools and calls a Gym Model Server for inference.

Capability level: **Evaluation**. The adapter returns the root session as Responses items and an observation bundle
with explicit gaps. Token IDs are not captured.

Goose's native subagents come from its built-in `summon` extension: the `delegate` tool runs a subagent in its own
session, synchronously or with `async: true` followed by `load(task_id)`. Several delegate calls in one turn run
concurrently. Delegates cannot delegate further (depth 1).

## Configure and run

[configs/goose_agent.yaml](configs/goose_agent.yaml) defines the agent defaults. For SWE-bench Pro, save this
composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/goose_agent/configs/goose_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: goose_agent
      resources_tool_transports: []
```

Supply a `policy_model` Gym Model Server, `policy_model_name`, and the `sandbox` provider configuration in
`model-provider.yaml`. The task sandbox must be able to reach the Model Server.

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml --allow-unsupported-pairing

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent goose_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

| Setting | Default | Purpose |
|---|---:|---|
| `goose_version` | `1.54.0` | Release whose archive checksums are pinned in the adapter. Other values are refused. |
| `model` | `${policy_model_name}` | `GOOSE_MODEL`, the model name Goose sends. |
| `timeout` | `3600` | Seconds per activation, enforced by the sandbox's process supervisor. Task `agent_timeout_sec` metadata can shorten it. |
| `max_turns` | `1000` | `goose run --max-turns` for the root. Delegates use Goose's own limit. |
| `delegate` | `true` | `false` disables `summon` (`delegate` and `load`), for a solo control. |
| `subagent_model` | `null` | `GOOSE_SUBAGENT_MODEL` for delegates; `null` reuses `model`. |
| `context_limit` | `null` | `GOOSE_CONTEXT_LIMIT`: the served model's context window, for Goose's compaction. |
| `system_prompt` | `null` | Prepended to the request's system text. |
| `sandbox_install_timeout_seconds` | `900` | Bound on session setup, including installation. |
| `session_close_timeout_seconds` | `60` | Bound on each supervised close phase. |
| `sandbox_provider`, `sandbox_config` | `null`, `{}` | Agent-owned sandbox, only when Resources supplies no `SandboxAccess`. |

## Runtime and sandbox setup

Session setup installs the pinned static (musl) release for the sandbox's architecture (`uname -m`: `x86_64` or
`aarch64`) under `/tmp/nemo-gym-goose-runtime-1.54.0`. The agent server downloads the GitHub release archive once into
its asset cache, checks the SHA-256 pinned in `sandbox.py`, and uploads it; the sandbox checks the checksum again,
extracts it, checks the version Goose reports, and reuses a runtime that already reports it. The sandbox needs only
bash, `sha256sum`, `tar` and `gzip`, and no internet access; the agent server needs access to GitHub releases.

Goose runs in `SandboxAccess.workdir` through the shared `harness_runner.sh` under the process supervisor. Its
config, sessions and state live under `GOOSE_PATH_ROOT` in a private session directory, so tool shells keep the
task's `HOME` and `XDG_*` directories. `SandboxAccess.env` is forwarded, except `GOOSE_*` and `OPENAI_*` variables,
which could redirect Goose's configuration or model calls. Close removes only the session directory; the Resources
Server destroys a borrowed sandbox, and the agent stops a sandbox it owns.

## Requests, results and outcomes

The request's system or developer message and `instructions` form Goose's system prompt (`--system`, at most
128 KiB), and the user message is the task, passed on stdin. Request `temperature` is passed as `GOOSE_TEMPERATURE`.
Other request fields, such as `max_output_tokens`, `top_p` and `tools`, are refused with HTTP 422, as are sessionless
requests and `/run`.

The response output is the root session converted to Responses items (reasoning, messages, function calls and their
outputs). Goose's own messages (`userVisible: false`, such as compaction notices) are not output. Response metadata
records `goose_exit_code`, `goose_stop_reason`, `goose_subagents` and `goose_max_concurrent_subagents`.

| Goose 1.54.0 ending | Response |
|---|---|
| Finished | `completed` |
| Supervisor deadline, with model output | `incomplete` (verified); without any model output, the activation fails |
| `--max-turns` notice, or killed before the deadline (exit 137, usually out of memory in the task cgroup) | `incomplete` (verified) |
| Completion cut at the output or context limit | `incomplete`, `max_output_tokens` |
| Model calls failed at the Model Server (network, 5xx, rate limit, rejected credentials, missing model), in the root or a delegate | `failed`, `model_transport_error` (retryable) |
| The model kept returning empty completions | `failed`, `agent_invalid_output` |
| Another Goose error notice, or a nonzero exit | `failed`, `agent_run_error` |
| Goose never started, left no session store, or an extension failed to start | The activation raises |

## Limitations

- Usage is `null`: Goose's usage ledger is not tied to individual model calls. The observation bundle declares
  `usage_unavailable`, `model_call_ownership_unavailable` (Goose sends no per-call session header),
  `harness_tool_timing_unavailable` and `child_invocations_unavailable`.
- Required tool grants are refused at seed; optional grants are skipped and reported. A `SandboxAccess` user override
  is refused.
- One activation per session; there is no host execution path.
- The solo control removes `delegate` and `load`, but Goose's extension manager stays enabled, so a model could in
  principle re-enable `summon` at run time. Delegates started with `async: true` are cancelled if the root ends
  without collecting them with `load`.

## Tests

```bash
python -m pytest responses_api_agents/goose_agent/tests -q -o addopts=''
```

The lifecycle tests run the shared process supervisor and runner on Linux with a fake Goose CLI.
