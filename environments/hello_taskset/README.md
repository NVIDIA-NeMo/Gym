# Hello Taskset

Hello Taskset introduces a taskset: a named collection of related tasks that you can run together. Here, each task supplies its own file path and message, while all tasks share one instruction template and verifier.

The environment contains two tasks:

- **Hello** creates `/workspace/hello-gym.txt` with `Hello from NeMo Gym!`.
- **Goodbye** creates `/workspace/goodbye-gym.txt` with `Goodbye from NeMo Gym!`.

Both tasks use the same instruction template and environment-level verifier.

Their JSONL rows contain only the path and expected content that differ.

## Run it

```bash
export ANTHROPIC_API_KEY="your-api-key"

gym eval run \
  --environment hello-taskset \
  --taskset example \
  --agent claude_code_agent \
  --model claude-sonnet-4-6
```

> [!NOTE]
> This is the target interface. Environment loading and execution are not implemented yet.

## How it works

For each JSONL row, Gym:

1. Validates `task_data` with the Pydantic model in `task.py`.
2. Renders `instruction.md` with that data.
3. Creates a clean workspace from `runtime/Dockerfile`.
4. Runs the agent in `/workspace`.
5. Passes the selected verifier input to the shared `verifier.py`.
6. Returns reward `1.0` for success or `0.0` otherwise.

This data-authored form is useful when tasks are structurally uniform. When individual tasks need different policies, use task-local verifiers.

## Related examples

- [Hello World](../hello_world/README.md) — Create the smallest single-task environment.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md) — Reuse verification code while keeping task-specific policy.
- [Hello MCP Tool](../hello_mcp_tool/README.md) — Give an agent an additional tool over MCP.
