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
> This is the target one-command interface. The dependent runtime prototype now loads, validates, renders, and verifies this taskset; connecting that runtime directly to `--environment` remains future work.

## How it works

For each JSONL row, Gym:

1. Validates `task_data` with the Pydantic model in `task.py`.
2. Renders `instruction.md` with that data.
3. Creates a clean workspace from `runtime/Dockerfile`.
4. Runs the agent in `/workspace`.
5. Passes the selected verifier input to the shared `verifier.py`.
6. Returns reward `1.0` for success or `0.0` otherwise.

This data-authored form is useful when tasks are structurally uniform. When individual tasks need different success
criteria, use task-local verifiers.

## Why JSONL?

JSONL keeps large collections of uniform tasks compact, streamable, and easy to generate, shard, filter, or publish as datasets. Each row contains only the values that vary; the instruction, runtime, data model, and verifier remain shared.

Use task directories instead when individual tasks need their own instructions, verifier logic, fixtures, repositories,
tests, or other assets. [Hello Verifier Reuse](../hello_verifier_reuse/README.md) demonstrates that directory-based
form.

## Related examples

- [Hello World](../hello_world/README.md) — Create the smallest single-task environment.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md) — Reuse verification code while keeping task-specific
  verifier logic.
- [Hello MCP Tool](../hello_mcp_tool/README.md) — Give an agent an additional tool over MCP.
