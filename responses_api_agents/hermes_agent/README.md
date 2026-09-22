# Hermes Agent

## Native sandbox sessions

EnvironmentServer creates the Resources session first, then passes its `SandboxAccess` to Hermes.
Hermes installs its pinned runtime and runs inside that task sandbox; Resources owns verification
and sandbox teardown. Use `enabled_toolsets: [terminal]`, one agent-server worker, and a Linux
sandbox with PTY process support. Existing calls without an agent session keep the local path below.

Each native session accepts one activation. Repeated activations and stale session cookies return
409. Close stops the active invocation and requires a supervisor receipt confirming that tool
descendants have exited before verification can proceed. Unknown launch outcomes or missing
receipts fail close. Successful close receipts remain available for
`session_close_retry_window_seconds` (default: 300 seconds), measured from successful cleanup.
Other sessions cannot evict them early, and retries do not extend expiry. Set the window to cover
the caller's full close-retry horizon, including response timeouts and backoff. After expiry, close
returns 409; the stale cookie still cannot activate the host path. Expired receipts are pruned on
session seed/close activity. Receipt memory scales with the close rate and window; this is not
durable storage across agent-server restarts.

Native input may be a string or text history ending in a user message, optionally starting with
a system message. Configured system text, request `instructions`, and input system text are combined.
Request `max_output_tokens` and `temperature` override their configured defaults; the token limit
applies to each model call, as with the existing `max_tokens` config. Unsupported input modes,
sampling fields, and required Resources tools return 422 rather than being silently ignored.

Usage aggregation is still a follow-up: the final response currently reports zero usage even when
the trajectory contains token IDs. Cleanup is cooperative, not isolation against hostile task code.
Resources/provider expiry is still needed after an agent-server crash.

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
| `system_prompt` | `null` | passed as `system_message` to `run_conversation`; falls back to any system item in `body.input` |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->
