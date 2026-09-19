# Hello World

Hello World is the smallest NeMo Gym environment and shows how to create and run your first task. It asks an agent to create one text file, then checks the result with a task-local verifier.

## Run it

```bash
export ANTHROPIC_API_KEY="your-api-key"

gym eval run \
  --environment hello-world \
  --agent claude_code_agent \
  --model claude-sonnet-4-6
```

> [!NOTE]
> This is the target one-command interface. The dependent runtime prototype can already run this task with Hermes in local Docker, but connecting that runtime directly to `--environment` remains future work.

## How it works

- `instruction.md` tells the agent what to do.
- `verifier.py` defines success for this task.
- `runtime/Dockerfile` defines the workspace where the agent works.
- `environment.yaml` connects those pieces.

The `episode_protocol` field tells Gym what kind of interaction to run for each task. This example uses the built-in single-agent protocol.

Hello World contains one task, so Gym runs it automatically. A taskset is a named collection of similar tasks that uses a `tasksets/` folder and the `--taskset` option.

The singleton task stays at the environment root to keep the first example small. When tasks need different instructions, verifier logic, or assets, they can move into task directories as shown by Hello Verifier Reuse. When many tasks share those definitions and differ only in data, use JSONL as shown by Hello Taskset.

## Related examples

- [Hello Taskset](../hello_taskset/README.md) — Run a named collection of similar tasks.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md) — Reuse verification code while keeping task-specific verifier logic.
- [Hello MCP Tool](../hello_mcp_tool/README.md) — Give an agent an additional tool over MCP.
