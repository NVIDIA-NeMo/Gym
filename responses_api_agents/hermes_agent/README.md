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

Runs [hermes-agent](https://github.com/NousResearch/hermes-agent) as a managed child process of a NeMo Gym
agent server. Each rollout uses an isolated Hermes home and the `run_agent.AIAgent` entrypoint from the upstream CLI
runtime. It can be used for evaluation or for training with the harness.

## Setup

`hermes-agent` is pinned to the upstream Hermes Agent 0.20.5 release (`v2026.8.19`, commit
`fcbd1076a93841fa88855acce810e342a5b78101`). On first startup, Gym downloads the checksum-pinned official Hermes
installer and uses its `repository`, `venv`, and `python-deps` stages to create a cached, self-contained runtime under
`~/.cache/nemo-gym/hermes-agent/0.20.5`. This keeps Hermes's dependency versions, including its newer OpenAI SDK,
isolated from the Gym agent server. Gym finishes installation with a frozen sync against the release's `uv.lock`.

### Runtime and observability boundary

The Gym agent server runs on Python 3.13, while the official Hermes installer creates a separate Python 3.11
environment with Hermes's own dependency versions. Importing `AIAgent` into the server process would mix these
incompatible environments, so `runner.py` executes Hermes with its managed interpreter.

Hermes observability hooks must wrap the live `AIAgent` inside that child process. `raw_observability.py` therefore
collects dependency-free, JSON-serializable events there, and the parent process converts them into Gym's canonical
`AgentObservationBundle` models in `observability.py`. CLI-based agents that already emit structured artifacts can
perform that conversion entirely in the parent process and do not require this two-layer bridge.

Each rollout receives its own `HERMES_HOME`. Memory, background review, session persistence, and context-file loading
are intentionally disabled and are not configurable. The agent server treats every rollout as stateless and deletes
its temporary Hermes home afterward. Sampling and chat-template settings are passed through Hermes's
`request_overrides`, and agent/tool activity is returned through Gym's rollout observability contract.

## Training

Upstream Hermes does not return Gym's token fields inline. To collect a Hermes rollout for training, explicitly set
`token_id_capture: true` on the Hermes agent and enable Gym's global `token_id_capture` block in the configuration
passed to both the servers and rollout collection:

```yaml
hermes_agent:
  responses_api_agents:
    hermes_agent:
      token_id_capture: true

token_id_capture:
  enabled: true
  dir: /absolute/shared/path/to/token-captures
```

The directory must be visible at the same absolute path to the Gym Model Server and rollout collector. A custom
`token_id_capture.sink` can replace the directory for distributed deployments.

Hermes always calls the configured Gym `model_server`. Token capture is an optional training mode on that route; it
does not select or change the model server or its inference provider. Successful reconstruction requires each model
response to contain exact `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs`. Gym masks the
training sample when those fields are unavailable rather than reconstructing them approximately. Model-server
sampling overrides remain authoritative for on-policy training. Evaluation does not require token capture.

## Resources server compatibility

Hermes works with verifier-only resources servers without additional configuration. When a resources server uses
`expose_tools_over_mcp: true`, its `/seed_session` response includes a rollout-specific HTTP MCP endpoint and signed
session header. The Hermes agent adds that server to the current rollout's temporary `HERMES_HOME`, discovers its tools
before constructing `AIAgent`, and deletes the configuration with the rest of the rollout home afterward. Concurrent
rollouts therefore do not share MCP session credentials.

Resources-server MCP tools coexist with Hermes's built-in toolsets (terminal, file, code execution, web, and others).
Hermes exposes each MCP tool to the model as `mcp__<server>__<tool>`. Resources servers that do not return MCP metadata
retain the verifier-only behavior.

When the seed metadata advertises the server's tool names, the agent also records each executed MCP call's raw
`server_name` and `tool_name` in `mcp_tool_call_provenance`, keyed by the Responses API `call_id`. Verification and
reverification can therefore use the canonical identity without parsing Hermes's model-facing alias. Built-in tools
and MCP calls from other servers are not attributed to the rollout resources server.

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
| `token_id_capture` | `false` | opts this opaque external harness into capture when run-level `token_id_capture.enabled` is also true |
| `max_turns` | `30` | maps to `AIAgent.max_iterations` |
| `concurrency` | `32` | max simultaneous `run()` calls |
| `temperature` | `1.0` | sampling temperature passed to `AIAgent` |
| `terminal_backend` | `local` | selects Hermes's internal tool-execution backend; Gym supports and tests only `local` |
| `terminal_timeout` | `60` | sets `TERMINAL_TIMEOUT` for each isolated Hermes child; per-command wall-clock seconds |
| `system_prompt` | `null` | passed as `system_message` to `run_conversation`; falls back to any system item in `body.input` |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->

`terminal_backend` controls Hermes's own terminal, file, and code tools; it does not move the Hermes agent runtime.
With `local`, both the runtime and those tools execute on the agent-server host. Non-local values require the
corresponding upstream Hermes backend to be configured separately; Gym does not provision or test those backends.
