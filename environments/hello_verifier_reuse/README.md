# Hello Verifier Reuse

Hello Verifier Reuse combines reusable verifier utilities with task-specific policy. It shows two ways to reuse verification code without forcing every task to share the same policy.

The `default` taskset contains two tasks:

- **Shared verifier (`shared-verifier`)** references a complete verifier from `nemo_gym.verifiers`.
- **Custom policy (`custom-policy`)** owns its verifier but imports a shared file-reading helper.

## Run it

```bash
export ANTHROPIC_API_KEY="your-api-key"

gym eval run \
  --environment hello-verifier-reuse \
  --taskset default \
  --task custom-policy \
  --agent claude_code_agent \
  --model claude-sonnet-4-6
```

> [!NOTE]
> This is the target one-command interface. The dependent runtime prototype now loads these directory-based tasks and dispatches each selected verifier; connecting that runtime directly to `--environment` remains future work.

## How it works

Reusable libraries are a good home for difficult evidence extraction: trajectory parsing, command detection, filesystem inspection, and output decoding. Individual tasks should retain the policy that interprets that evidence.

The custom policy imports `read_text` from `nemo_gym.verifiers.files`, then adds its own requirement that the output be uppercase. A real evaluation can use the same pattern with helpers from an environment-local module or an installed Python package.

Use an environment-level verifier only when the tasks are genuinely uniform.

## Why task directories?

The tasks use directories because each one owns different instructions and verifier policy. A directory can also hold task-specific fixtures, repositories, tests, or other assets while remaining reviewable as one self-contained task.

Use JSONL instead when many tasks share the same structure and differ only in data values. [Hello Taskset](../hello_taskset/README.md) demonstrates that data-driven form.

## Related examples

- [Hello World](../hello_world/README.md) — Create the smallest single-task environment.
- [Hello Taskset](../hello_taskset/README.md) — Run a named collection of similar tasks.
- [Hello MCP Tool](../hello_mcp_tool/README.md) — Give an agent an additional tool over MCP.
