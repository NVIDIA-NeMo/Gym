# Goose Agent

Runs [Goose](https://github.com/aaif-goose/goose) **1.54.0** (Apache-2.0) headless with `goose run` inside a Gym
task sandbox. The Environment Server seeds the Resources and agent sessions, activates Goose once, closes the agent,
verifies the task, and closes Resources. Goose runs its own tools and calls a Gym Model Server for inference.

Capability level: **Evaluation**. The adapter returns the root session as Responses items and an observation bundle
with explicit gaps.

Model calls already use the rollout's capture route, but no training capture run has validated this harness yet.
The harness conformance probes are not run at the Evaluation level.

Goose's native subagents come from its built-in `summon` extension: the `delegate` tool runs a subagent in its own
session, synchronously or with `async: true` followed by `load(task_id)`. Several delegate calls in one turn run
concurrently. Delegates cannot delegate further (depth 1). `GOOSE_SUBAGENT_PROVIDER` and `GOOSE_SUBAGENT_MODEL` pin
delegates to the same Gym Model Server and model, whatever provider or model the policy passes to `delegate`.

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
The SWE-bench Pro pairing is undeclared (this agent is not in that Resources server's `allowed_agents`), so
`gym env start` needs `--allow-unsupported-pairing`.

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
| `delegate` | `true` | `false` disables `summon` (`delegate` and `load`) and `extensionmanager`, for a solo control. |
| `context_limit` | `null` | `GOOSE_CONTEXT_LIMIT`: the served model's context window, for Goose's compaction. |
| `system_prompt` | `null` | Prepended to the request's system text. |
| `sandbox_install_timeout_seconds` | `900` | Bound on session setup, including installation. |
| `session_close_timeout_seconds` | `60` | Bound on each supervised close phase. |
| `sandbox_provider`, `sandbox_config` | `null`, `{}` | Agent-owned sandbox, only when Resources supplies no `SandboxAccess`. |
| `network_access` | `inherit` | Egress of an agent-owned sandbox; `model_only` allows only the model server's host. |

## Runtime and sandbox setup

Session setup installs the pinned static (musl) release for the sandbox's architecture (`uname -m`: `x86_64` or
`aarch64`) under `/tmp/nemo-gym-goose-runtime-1.54.0`. The agent server downloads the GitHub release archive once into
its asset cache, checks the SHA-256 pinned in `sandbox.py`, and uploads it. On every seed the sandbox checks the
checksum again, extracts it, checks the version Goose reports and publishes it, so a binary already at the runtime path
is never trusted. The sandbox needs only bash, `sha256sum`, `tar`, `gzip` and `timeout`, and no internet access. The
agent server needs access to GitHub releases.

Goose needs CA certificates to construct its HTTP client, even when the model endpoint uses HTTP. When the
task image has no trust store (no `SSL_CERT_FILE` or `SSL_CERT_DIR`, and none of the standard bundle files such
as `/etc/ssl/certs/ca-certificates.crt`), setup uploads Certifi's public CA bundle into the private session
directory and selects it with `SSL_CERT_FILE`. Goose's tool shells inherit that variable, so an image with
its own trust store is left alone: the task, and any certificate the agent adds to it, keep working as they
would under another harness. Certificate verification stays enabled, and an explicit
`SandboxAccess.env.SSL_CERT_FILE` is preserved for endpoints that use a custom trust store.

Goose runs in `SandboxAccess.workdir` through the shared `harness_runner.sh` under the process supervisor. Its
config, sessions and state live under `GOOSE_PATH_ROOT` in a private session directory, so tool shells keep the
task's `HOME` and `XDG_*` directories. `SandboxAccess.env` is forwarded, except `GOOSE_*`, `OPENAI_*`, `OTEL_*` and
`LANGFUSE_*` variables, which could redirect Goose's configuration, model calls or traces. Close removes only the session directory. The Resources
Server destroys a borrowed sandbox, and the agent stops a sandbox it owns.

`network_access: model_only` limits an agent-owned sandbox's egress to the model server's host (every port). The
agent server uploads Goose, so setup needs no other host. The OpenSandbox policy is set when the sandbox is created, so
a session that borrows a task sandbox rejects any value other than `inherit`, as does a provider without network
policies or an explicit `sandbox_config.provider_options.network_policy`. Goose uses no tool grants, so
`model_and_tools` is rejected. Each refusal fails the seed with HTTP 422.

## Requests, results and outcomes

The request's system or developer message and `instructions` form Goose's system prompt (`--system`, at most
128 KiB), and the user message is the task, passed on stdin. Request `temperature` is passed as `GOOSE_TEMPERATURE`.
Other request fields, such as `max_output_tokens`, `top_p` and `tools`, are refused with HTTP 422, as are sessionless
requests and `/run`.

The response output is the root session converted to Responses items (reasoning, messages, function calls and their
outputs). Goose's own messages (`userVisible: false`, such as compaction notices) are not output. Response metadata
records `goose_exit_code`, `goose_stop_reason`, `goose_subagents` and `goose_max_concurrent_subagents`. A failed
response carries Goose's notice in `error`. Every model call sends `x-session-id` set to the rollout's capture key, the
invocation ID of the observation bundle.

| Goose 1.54.0 ending | Response |
|---|---|
| Finished, including after a delegate's error the root recovered from | `completed` |
| Supervisor deadline or rollout stop, with or without model output | `incomplete` (verified) |
| `--max-turns` notice, or the context limit still exceeded after compaction | `incomplete` (verified) |
| Completion cut at the output or context limit | `incomplete`, `max_output_tokens` |
| The root's model call failed with an HTTP error at the Model Server | `failed`, `model_request_error` with `status_code` |
| The root could not reach the Model Server | `failed`, `model_transport_error` |
| The model kept returning empty completions | `failed`, `agent_invalid_output` |
| Another Goose error notice, or a nonzero exit (including a bare 137) | `failed`, `unknown` |
| Goose never started, left no session store, or an extension failed to start | The activation raises |

## Limitations

- Usage is `null`: Goose's usage ledger is not tied to individual model calls. The observation bundle declares
  `usage_unavailable`, `harness_tool_timing_unavailable` and `child_invocations_unavailable`. Delegates' model calls
  carry the root's `x-session-id`.
- Required tool grants are refused at seed; optional grants are skipped and reported. A `SandboxAccess` user override
  is refused.
- One activation per session; there is no host execution path.
- Delegates started with `async: true` are cancelled if the root ends without collecting them with `load`.

## Tests

```bash
python -m pytest responses_api_agents/goose_agent/tests -q -o addopts=''
```

The lifecycle tests run the shared process supervisor and runner on Linux with a fake Goose CLI.
