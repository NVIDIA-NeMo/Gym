# Hello Verifier Reuse

Hello Verifier Reuse combines reusable verifier utilities with task-specific verifier logic. It shows two ways to reuse verification code without forcing every task to use the same success criteria.

The `default` taskset contains two tasks:

- **Shared verifier (`shared-verifier`)** references a complete verifier from `nemo_gym.verifiers`.
- **Custom verifier (`custom-verifier`)** owns its verifier but imports a shared file-reading helper.

## Run it

```bash
export MODEL_API_KEY="<api-key>"

gym eval run \
  --environment hello-verifier-reuse \
  --taskset default \
  --agent-type hermes_agent/borrowed_sandbox_openai_compatible \
  --model-type openai_model \
  --model <model-name> \
  --model-url <openai-compatible-url> \
  --model-api-key "$MODEL_API_KEY"
```

> [!NOTE]
> `--environment` materializes these directory-based tasks and generates the temporary runtime composition internally. The environment adapter dispatches the verifier selected by each task.

## How it works

Reusable libraries are a good home for difficult evidence extraction: trajectory parsing, command detection, filesystem inspection, and output decoding. Individual tasks can retain the verifier logic that interprets that evidence.

The custom verifier imports `read_text` from `nemo_gym.verifiers.files`, then adds its own requirement that the output be uppercase. A real evaluation can use the same pattern with helpers from an environment-local module or an installed Python package.

Use an environment-level verifier only when the tasks are genuinely uniform.

## Why task directories?

The tasks use directories because each one owns different instructions and verifier logic. A directory can also hold
task-specific fixtures, repositories, tests, or other assets while remaining reviewable as one self-contained task.

Use JSONL instead when many tasks share the same structure and differ only in data values. [Hello Taskset](../hello_taskset/README.md) demonstrates that data-authored form.

## Related examples

- [Hello World](../hello_world/README.md) — Create the smallest single-task environment.
- [Hello Taskset](../hello_taskset/README.md) — Run a named collection of similar tasks.
- [Hello MCP Tool](../hello_mcp_tool/README.md) — Give an agent an additional tool over MCP.
