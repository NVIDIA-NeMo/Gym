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
| `temperature` | `null` | sampling temperature passed to `AIAgent`; request `temperature` overrides it, including `0.0` |
| `terminal_backend` | `local` | sets `TERMINAL_ENV` (process-global); `local`, `docker`, `daytona`, `modal`, `ssh` |
| `terminal_timeout` | `60` | sets `TERMINAL_TIMEOUT` (process-global); per-command wall-clock seconds |
| `sandbox_provider` | `null` | named provider used to create an agent-owned sandbox when Resources does not supply `sandbox_access` |
| `sandbox_config` | `{}` | `SandboxSpec` fields used with `sandbox_provider`; ignored when Resources supplies a sandbox |
| `sandbox_runner_timeout_seconds` | `21600` | bounds one sandbox activation; the episode deadline still applies |
| `system_prompt` | `null` | joined with request `instructions` and the first input system message, in that order; appended to Hermes' built-in prompt |
| `session_close_retry_window_seconds` | `300` | session close receipt retention from successful cleanup; retries do not extend expiry |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->

Host and sandbox execution use the same request rules. User tasks remain user messages.
Request `model` must match the configured model. Unsupported non-default controls return
HTTP 422, including `top_p` (even `1.0`), `store`, `service_tier`, and request metadata.
`max_output_tokens` also returns 422: Hermes does not enforce a total response token budget.
Config `max_tokens` limits each model call. Only text input is supported; `developer` messages
are rejected.

Compatibility: the host path previously let config `system_prompt` replace the dataset's
system message and ignored request `temperature`. It now combines the prompts and honors
the request temperature, just like sandbox execution. These changes can affect scores.
Remove unsupported fields that older versions silently ignored.

For SWE-bench Pro, use [`hermes.yaml`](../../benchmarks/swebench/pro/hermes.yaml).
See [Evaluate SWE-bench Pro with Hermes](../../fern/versions/latest/pages/evaluation-tutorials/hermes-swe-bench-pro.mdx)
for task preparation, Environment Server configuration, evaluation commands, and session limits.

## Sandbox-mode requirements

Sandbox sessions live in the memory of the worker that seeded them, so seeding a session requires `num_workers: 1`. Calling the agent's `/run` directly keeps no session and still supports several workers.

Each sandbox session installs the Hermes version pinned in `requirements.txt`, the same one this server runs, at seed time unless it already imports from `/tmp/nemo-gym-hermes-runtime-<commit>/venv`, for example because the image bakes it in or an earlier session in the same sandbox installed it. Installing needs outbound access to GitHub and the Python package index. Hermes calls the Model Server directly from the sandbox, so the sandbox must also reach the Model Server at its configured host and port. Its image must also match the host CPU architecture and C library because the host's `uv` executable is copied into the sandbox.

Sessions use MCP tool grants and reject other required grants. Each granted MCP server is added to that session's Hermes configuration, so the sandbox must reach it at the granted URL, usually the Resources Server's `/mcp` endpoint. An activation fails before its first model call when a required server does not connect. Hermes names MCP tools `mcp_<server>_<tool>`; the response reports them as `mcp__<server>__<tool>`, the form Gym strips before verification, while captured model calls keep Hermes' names. The session token in each grant is readable inside the sandbox and gives access only to that episode's tools.

The Hermes runner and the model's terminal tool execute as the same user in the same sandbox. The host reads the final result, including token IDs, from `/tmp/nemo-gym-hermes-sessions/<session-id>/output.json`; commands issued by the model can also write that file. Sandbox mode is suitable for evaluation, but it must not be used to produce RL training data until results are returned through a channel the model cannot modify.
