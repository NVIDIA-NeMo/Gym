# Hermes Agent

# Quick start

## Create env.yaml in Gym/

```
policy_base_url: https://api.openai.com/v1
policy_api_key: sk...
policy_model_name: gpt-4o
```

## Launch nemo gym servers

```bash
gym env start \
    --config environments/hermes_math/config.yaml \
    --model-type openai_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent hermes_math_agent \
    --input environments/hermes_math/data/example.jsonl \
    --output hermes_agent_rollout.jsonl \
    --limit 1
```

Example math rollouts are in `environments/hermes_math/data/example_rollouts.jsonl`.

Example training reward for small multi environment test is shown [here](https://github.com/NVIDIA-NeMo/Gym/pull/1033#issuecomment-4399509664).

## Description

Runs [hermes-agent](https://github.com/NousResearch/hermes-agent) in a nemo gym agent server via the `run_agent.AIAgent` entrypoint, which matches the hermes-agent CLI and user experience. Can be used for benchmarks with hermes agent, or training in the harness.

## Setup

`hermes-agent` is pinned in `requirements.txt` to a fork branch with patches for token id tracking, chat template, and sampling parameters needed for training.

For agent integrations like this, the agent must point at Gym's model server, it must include prompt and generation token id in requests for Nemo RL and other trainer integration on policy token id correction, it must not override sampling parameters like temperature and top p, and it must not do non-monotonic things like dropping past reasoning content or context compaction.

## Resources server compatibility

Works with any resources server based verifier, but does not work for resources server tools or other endpoints out of the box. Hermes Agent ships its own toolset (terminal, file, code_execution, web, etc.), so it does not rely on tools defined in the dataset. It may work with Gymnasium style resources servers, though. In testing, only the resources server's task data and `verify` are used. This means existing benchmarks (math, code, reasoning_gym, mcqa, instruction_following, ...) can be used as-is by adding a `<server>_hermes_agent` config.

## Configuration example

```yaml
hermes_agent:
  responses_api_agents:
    hermes_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_verifier
      model_server:
        type: responses_api_models
        name: policy_model
      model: served-model-name
      enabled_toolsets: [terminal, file, code_execution]
      max_turns: 30
      concurrency: 32
      temperature: 1.0
      sandbox_provider: sandbox
      sandbox_config:
        image: my-agent-image
        ttl_s: 3600
        workdir: /workspace
      system_prompt: |
        your system prompt here.
```

| field | default | description |
|-------|---------|-------------|
| `enabled_toolsets` | `null` (all) | forwarded to `AIAgent(enabled_toolsets=...)` |
| `disabled_toolsets` | `null` | forwarded to `AIAgent(disabled_toolsets=...)` |
| `model` | `null` | served model id; defaults to `model_server.name` for backward compatibility |
| `max_turns` | `30` | maps to `AIAgent.max_iterations` |
| `concurrency` | `32` | max simultaneous `run()` calls |
| `temperature` | `1.0` | sampling temperature passed to `AIAgent` |
| `terminal_backend` | `local` | sets `TERMINAL_ENV` (process-global); `local`, `docker`, `daytona`, `modal`, `ssh` |
| `terminal_timeout` | `60` | sets `TERMINAL_TIMEOUT` (process-global); per-command wall-clock seconds |
| `sandbox_provider` | `null` | named provider used to create an agent-owned sandbox when Resources does not supply `sandbox_access` |
| `sandbox_config` | `{}` | `SandboxSpec` fields used with `sandbox_provider`; ignored when Resources supplies a sandbox |
| `sandbox_runner_timeout_seconds` | `21600` | bounds one sandbox activation; the episode deadline still applies |
| `system_prompt` | `null` | passed as `system_message` to `run_conversation`; falls back to any system item in `body.input` |
| `session_close_retry_window_seconds` | `300` | native-session close receipt retention from successful cleanup; retries do not extend expiry |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->

Native EnvironmentServer sessions run Hermes inside a sandbox. They borrow the Resources-owned
task sandbox when supplied, or create an agent-owned sandbox from `sandbox_provider` and
`sandbox_config`. SWE-bench Pro uses the borrowed path. Use one agent-server worker and a Linux
sandbox with exec support; the benchmark recipe enables `[terminal]`.
Each session runs once and must confirm process cleanup before verification. Close receipts are
process-local; configure the retry window to cover response timeouts and backoff. Other sessions
cannot evict receipts early, and expired sessions return 409 without falling back to the host.
Native requests support text input and instructions; output limits apply per model call, and
unsupported settings return 422. Response usage is still zero pending aggregation support.
The iteration-limit summary uses the regular model-request formatter and settings, preserving
reasoning and tool history while keeping internal Responses IDs out of Chat Completions.
Summary request failures propagate as activation failures with partial observations.


For native TB4 collection, compose the generic
[episode profile](../../benchmarks/terminal_bench_4/episode.yaml) with this agent:

```bash
gym env start --config benchmarks/terminal_bench_4/episode.yaml \
  --agent hermes_agent --config model-provider.yaml
```

The existing SWE-bench Pro `hermes.yaml` profile retains its upstream behavior.
Use the same composition when starting servers and collecting.

EnvironmentServer assigns session IDs before seeding and sends cleanup after a lost
seed response. Identical retries share a session; mismatched requests and closed IDs
are rejected. Abandoned sessions expire after `session_lifetime_seconds` (default 21600);
failed cleanup retains its handle for a retry. Close receipts expire separately from
closed-ID tombstones, which remain for at least the lifetime/retry horizon.


## Sandbox-mode requirements

Sandbox sessions live in the memory of the worker that seeded them, so seeding a session requires `num_workers: 1`. Calling the agent's `/run` directly keeps no session and still supports several workers.

Each sandbox session installs the Hermes version pinned in `requirements.txt`, the same one this server runs, at seed time unless it already imports from `/tmp/nemo-gym-hermes-runtime-<commit>/venv`, for example because the image bakes it in or an earlier session in the same sandbox installed it. Installing needs outbound access to GitHub and the Python package index. Hermes calls the Model Server directly from the sandbox, so the sandbox must also reach the Model Server at its configured host and port. Bootstrap probes the sandbox's architecture and libc, then uploads matching pinned uv and Python archives. The task image needs bash, uname, and tar; its architecture can differ from the agent server's. Session HOME and caches stay inside the adapter directory. Optional resource task context supplies the execution user, deadline, skills, and sandbox-local MCP connections.

Sessions use MCP tool grants and reject other required grants. Each granted MCP server is added to that session's Hermes configuration, so the sandbox must reach it at the granted URL, usually the Resources Server's `/mcp` endpoint. An activation fails before its first model call when a required server does not connect. Hermes names MCP tools `mcp_<server>_<tool>`; the response reports them as `mcp__<server>__<tool>`, the form Gym strips before verification, while captured model calls keep Hermes' names. The session token in each grant is readable inside the sandbox and gives access only to that episode's tools.

The Hermes runner and the model's terminal tool execute as the same user in the same sandbox. The host reads the final result, including token IDs, from `/tmp/nemo-gym-hermes-sessions/<session-id>/output.json`; commands issued by the model can also write that file. Sandbox mode is suitable for evaluation, but it must not be used to produce RL training data until results are returned through a channel the model cannot modify.
