# Hello MCP Tool

Hello MCP Tool gives an agent one additional tool over the Model Context Protocol (MCP), a standard way for an agent to discover and call external capabilities.

The task asks the agent to call `get_greeting` and write the returned greeting to its workspace. The verifier requires both the recorded tool call and the resulting file.

## Run it

```bash
export ANTHROPIC_API_KEY="your-api-key"

gym eval run \
  --environment hello-mcp-tool \
  --agent claude_code_agent \
  --model claude-sonnet-4-6
```

> [!NOTE]
> This is the target interface. Environment loading and execution are not implemented yet.

## How it works

- `tools/server.py` defines `get_greeting` with the MCP Python SDK.
- `environment.yaml` declares the MCP server and how Gym starts it.
- `pyproject.toml` declares the additional `mcp` dependency.
- `verifier.py` checks Gym's recorded tool calls and the workspace result.

Gym starts the MCP server, gives its connection information to the selected agent, records calls in the attempt, and stops the server with the environment session.

The other `hello-*` environments do not need `pyproject.toml` because they introduce no dependencies beyond Gym itself. Add one when an environment imports an additional Python package or needs independent packaging metadata.

## Related examples

- [Hello World](../hello_world/README.md) — Create the smallest single-task environment.
- [Hello Taskset](../hello_taskset/README.md) — Run a named collection of similar tasks.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md) — Reuse verification code while keeping task-specific policy.
