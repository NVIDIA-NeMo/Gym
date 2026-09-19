# Hello World

Hello World is the smallest NeMo Gym environment and shows how to create and run your first task. It asks an agent to create one text file, then checks the result with a task-local verifier.

## Run it

```bash
export MODEL_API_KEY="<api-key>"

gym eval run \
  --environment hello-world \
  --agent-type hermes_agent/borrowed_sandbox_openai_compatible \
  --model-type openai_model \
  --model <model-name> \
  --model-url <openai-compatible-url> \
  --model-api-key "$MODEL_API_KEY"
```

The `borrowed_sandbox_openai_compatible` profile enables Hermes's terminal tool while omitting
provider-specific chat-template arguments that standard OpenAI-compatible endpoints may reject.

> [!NOTE]
> `--environment` now loads `environment.yaml`, materializes its task internally, builds the local Docker runtime, and generates the temporary server composition. The environment does not need a `config.yaml`, prepared JSONL file, or `prepare.py`.

## How it works

- `instruction.md` tells the agent what to do.
- `verifier.py` defines success for this task.
- `runtime/Dockerfile` defines the workspace where the agent works.
- `environment.yaml` connects those pieces.

Hello World contains one task, so Gym runs it automatically. A taskset is a named collection of similar tasks that uses a `tasksets/` folder and the `--taskset` option.

The singleton task stays at the environment root to keep the first example small. When tasks need different instructions, verifier policies, or assets, they can move into task directories as shown by Hello Verifier Reuse. When many tasks share those definitions and differ only in data, use JSONL as shown by Hello Taskset.

## Related examples

- [Hello Taskset](../hello_taskset/README.md) — Run a named collection of similar tasks.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md) — Reuse verification code while keeping task-specific policy.
- [Hello MCP Tool](../hello_mcp_tool/README.md) — Give an agent an additional tool over MCP.
