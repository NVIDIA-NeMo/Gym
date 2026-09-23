# Hello MCP Tool

Hello MCP Tool lets you give an agent a tool it needs for a task without changing the agent itself.

## Concept

A task can declare required tools through the Model Context Protocol (MCP). NeMo Gym checks that the selected agent and sandbox can provide those tools before the run starts. MCP gives tasks a standard way to describe and connect them.

## In this example

- `task.toml` declares the `hello-tools` MCP server as a task requirement.
- `environment/tools/server.py` implements a `get_greeting` tool.
- `environment/Dockerfile` installs the server and its dependencies in the task environment.
- `instruction.md` asks the agent to use the tool and save its result.
- `tests/test.sh` checks the saved result.

## Run it

```bash
export OPENAI_API_KEY="<api-key>"

gym eval run environments/hello_mcp_tool \
  --agent opencode \
  --model openai/gpt-5 \
  --sandbox docker
```

The command uses OpenCode because it can work with files and tools in the task sandbox while keeping model choice separate. It uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

## Check the result

TODO: After runtime integration is available, add the literal successful CLI output and identify the exact result fields and artifacts a user should inspect to verify the score.

## Other examples

### Tasks and datasets

- [Hello World](../hello_world/README.md): Define one self-contained directory task.
- [Hello Dataset](../hello_dataset/README.md): Run multiple distinct tasks as one dataset.

### Python scoring

- [Hello Python Verifier](../hello_python_verifier/README.md): Write task-specific scoring logic in Python.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
