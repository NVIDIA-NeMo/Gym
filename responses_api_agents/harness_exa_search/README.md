# Harness Exa Search Environment

Runs a Gym agent harness in a sandbox with Exa search, then delegates grading
to the configured resources server. Benchmark-local configs are available for
Claude Code, Codex, OpenCode, Pi, Hermes, OpenClaw, Kilocode, Cline, Prime
Agent, Simple Strands, and NeMo Fabric DeepAgents. Validation depth varies by
harness, so use a one-task smoke test before a full run.

## Configuration

```yaml
responses_api_agents:
  harness_exa_search:
    model_server: {type: responses_api_models, name: policy_model}
    resources_server: {type: resources_servers, name: my_verifier}
    agent: claude_code
    agent_kwargs:
      model: nvidia/qwen/qwen3.8-27b
      max_turns: 30
      timeout: 900
      bare: true
      system_prompt: You must call an Exa MCP search tool before answering.
      claude_code_version: null
    image: nikolaik/python-nodejs:python3.13-nodejs22-slim
    setup_command: python3 -m pip install -q https://github.com/NVIDIA-NeMo/Gym/archive/4b30acc3eb3a316e5d6e0475e6959606c8512cbf.zip && npm install -g @anthropic-ai/claude-code
    sandbox_provider: sandbox
    sandbox_spec:
      ttl_s: 1800
    exa_api_key: <exa-api-key>
```

| Option | Description |
| --- | --- |
| `model_server` | Gym model server. |
| `resources_server` | Benchmark-specific verifier that receives the completed response. |
| `agent` | Agent name from Gym’s shared `harness_agent` registry, such as `claude_code`, `codex`, or `opencode`. |
| `agent_kwargs` | Arguments forwarded to the selected agent’s config class. Supported keys depend on the agent. |
| `image` | Sandbox base image containing Python and required system dependencies. |
| `python` | Python executable used to launch `agent_runner.py`. It defaults to `python3`. |
| `runtime_archive` | Optional AnySWE-style portable harness runtime. Gym uploads and unpacks it once per sandbox. |
| `setup_command` | Optional sandbox setup command run before the agent. |
| `sandbox_provider` | Gym sandbox provider reference. |
| `sandbox_spec` | Sandbox lifetime, resources, workdir, environment, and provider options. |
| `sandbox_model_base_url` | Optional model URL directly reachable from the sandbox. When omitted, endpoint-capable sandbox providers relay requests through the configured Gym model server. |
| `exa_api_key` | Exa credential passed only to the sandboxed runner for MCP configuration. |

For Claude Code, `agent_kwargs` are fields from `ClaudeCodeAgentConfig`.
`model` selects the hosted model, `max_turns` and `timeout` bound execution,
`bare` disables ambient local configuration, `system_prompt` requires web
research, and `claude_code_version` optionally pins the CLI version.

For remote sandboxes that cannot route to a cluster-private model address, the
agent opens a sandbox-local relay and forwards model traffic through the
configured Gym model server. This supports Messages, Responses, and Chat
Completions without an external tunnel.

See the complete, benchmark-specific config copies under
[`benchmarks/deepsearchqa/configs`](../../benchmarks/deepsearchqa/configs) and
[`benchmarks/widesearch/configs`](../../benchmarks/widesearch/configs). The
selected agent’s dependencies must be installed by its image or `setup_command`.
The Claude Code recipe above installs Gym and the Claude CLI inside a fresh
sandbox; it does not need a runtime archive from a cluster. `runtime_archive`
is an optional optimization for users who already have a compatible build.

## Benchmarks

- [`DeepSearchQA`](../../benchmarks/deepsearchqa/README.md) uses its
  single/set-answer resources server.
- [`WideSearch`](../../benchmarks/widesearch/README.md) uses its table-aware
  resources server.
