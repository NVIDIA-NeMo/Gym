# Harness Exa Search Environment

Runs a Gym agent harness in a sandbox with Exa MCP search, then delegates
grading to the configured resources server. Claude Code is the only harness
tested so far.

## Configuration

```yaml
responses_api_agents:
  harness_exa_search:
    model_server: {type: responses_api_models, name: policy_model}
    resources_server: {type: resources_servers, name: my_verifier}
    harness_module: responses_api_agents.claude_code_agent.app
    harness_class: ClaudeCodeAgent
    harness_config_class: ClaudeCodeAgentConfig
    harness_kwargs:
      model: nvidia/qwen/qwen3.8-27b
      max_turns: 30
      timeout: 900
      bare: true
      system_prompt: You must call an Exa MCP search tool before answering.
      claude_code_version: null
    image: <sandbox-image>
    setup_command: null
    sandbox_provider: sandbox
    sandbox_spec:
      ttl_s: 1800
    exa_api_key: <exa-api-key>
```

| Option | Description |
| --- | --- |
| `model_server` | Gym model server. |
| `resources_server` | Benchmark-specific verifier that receives the completed response. |
| `harness_module` | Import path containing the Gym harness implementation. |
| `harness_class` | Harness class instantiated by `agent_runner.py`. |
| `harness_config_class` | Pydantic config class used by the selected harness. |
| `harness_kwargs` | Arguments forwarded to that config class. Supported keys depend on the harness. |
| `image` | Sandbox image containing Python, the harness runtime, and any system dependencies. |
| `python` | Python executable used to launch `agent_runner.py`. It defaults to `python3`. |
| `setup_command` | Optional sandbox setup command run before the agent. |
| `sandbox_provider` | Gym sandbox provider reference. |
| `sandbox_spec` | Sandbox lifetime, resources, workdir, environment, and provider options. |
| `sandbox_model_base_url` | Optional model URL reachable from the sandbox. Otherwise Gym derives it. |
| `exa_api_key` | Exa credential passed only to the sandboxed runner for MCP configuration. |

For Claude Code, `harness_kwargs` are fields from `ClaudeCodeAgentConfig`.
`model` selects the hosted model, `max_turns` and `timeout` bound execution,
`bare` disables ambient local configuration, `system_prompt` requires web
research, and `claude_code_version` optionally pins the CLI version.

See [`configs/harness_exa_search_claude_code.yaml`](configs/harness_exa_search_claude_code.yaml)
for the complete config. The sandbox image must contain the selected harness
and its dependencies.

## Benchmarks

- [`DeepSearchQA`](../../benchmarks/deepsearchqa/README.md) uses its
  single/set-answer resources server.
- [`WideSearch`](../../benchmarks/widesearch/README.md) uses its table-aware
  resources server.
