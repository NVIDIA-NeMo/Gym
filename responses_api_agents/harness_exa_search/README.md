# Harness Exa Search Environment

Runs a Gym agent harness in OpenSandbox with Exa MCP search and an LLM judge
for single-answer and set-answer research tasks. Claude Code is the only
harness tested so far.

## Configuration

```yaml
responses_api_agents:
  harness_exa_search:
    model_server: {type: responses_api_models, name: policy_model}
    resources_server: {type: resources_servers, name: my_verifier}
    harness_module: responses_api_agents.claude_code_agent.app
    harness_class: ClaudeCodeAgent
    harness_config_class: ClaudeCodeAgentConfig
    image: ${oc.env:HARNESS_EXA_SEARCH_IMAGE}
    exa_api_key: ${oc.env:EXA_API_KEY,null}
```

See [`configs/harness_exa_search_claude_code.yaml`](configs/harness_exa_search_claude_code.yaml)
for the complete config. The sandbox image must contain the selected harness
and its dependencies.

## Benchmarks

- [`DeepSearchQA`](../../benchmarks/deepsearchqa/README.md) uses its
  single/set-answer resources server.
- WideSearch can reuse the sandboxed harness and Exa integration, but requires
  its own table-aware verifier.
