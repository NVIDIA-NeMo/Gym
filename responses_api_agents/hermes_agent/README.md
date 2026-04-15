# Hermes Agent

Runs [hermes-agent](https://github.com/NousResearch/hermes-agent)'s as a NeMo Gym agent server.

## Setup

Install hermes-agent alongside this repo:

```
responses_api_agents/hermes_agent/requirements.txt
```

The `requirements.txt` installs hermes-agent as a local editable package from the sibling directory.

## Configuration

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
      enabled_toolsets: [terminal, file, web_search]
      terminal_backend: local
      max_turns: 30
      concurrency: 32
      system_prompt: |
        Your system prompt here.
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled_toolsets` | `null` (all) | Hermes toolsets to enable |
| `disabled_toolsets` | `null` | Toolsets to exclude |
| `terminal_backend` | `local` | Terminal execution backend |
| `terminal_timeout` | `120` | Per-command timeout in seconds |
| `max_turns` | `30` | Max LLM calls per rollout |
| `concurrency` | `32` | Max simultaneous `run()` calls |
| `tool_pool_size` | `128` | Thread pool size for tool dispatch |
| `system_prompt` | `null` | Prepended to every rollout if set |

## Example configs

- `resources_servers/math_with_judge/configs/math_with_judge_hermes_agent.yaml` — math reasoning with terminal for computation
- `resources_servers/reasoning_gym/configs/reasoning_gym_hermes_agent.yaml` — pure reasoning, no tools

## Notes

Hermes tools and Gym resources-server tools are separate dispatch systems. If the data already defines tools (e.g. `execute_python` for `math_with_code`), hermes does not intercept those calls. Use benchmarks whose data has an empty tools list, or set `enabled_toolsets: []` for tool-free rollouts.
