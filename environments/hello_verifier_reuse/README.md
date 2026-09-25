# Hello Verifier Reuse

Hello Verifier Reuse combines reusable verifier utilities with task-specific verifier logic. It shows two ways to reuse verification code without forcing every task to use the same success criteria.

The `default` taskset contains two tasks:

- **Exact greeting (`exact-greeting`)** references a complete verifier from `nemo_gym.verifiers`.
- **Uppercase greeting (`uppercase-greeting`)** owns its task-local verifier but imports a reusable file-reading helper.

## Run it

```bash
export ANTHROPIC_API_KEY="your-api-key"

gym eval run \
  --environment hello-verifier-reuse \
  --agent claude_code_agent \
  --model claude-sonnet-4-6
```

> [!NOTE]
> This is the target one-command interface. The dependent runtime prototype now loads these directory-based tasks and dispatches each selected verifier; connecting that runtime directly to `--environment` remains future work.

## How it works

Reusable libraries are a good home for difficult evidence extraction: trajectory parsing, command detection, filesystem inspection, and output decoding. Individual tasks can retain the verifier logic that interprets that evidence.

`environment.yaml` points the `default` taskset at `tasks/`, so Gym selects it when `--taskset` is omitted. Gym
discovers each child directory containing `task.yaml`, using the directory name as its task ID.

The uppercase-greeting verifier imports `read_text` from `nemo_gym.verifiers.files`, then adds its own requirement that
the output be uppercase. A real evaluation can use the same pattern with helpers from an environment-local module or
an installed Python package.

## Why task directories?

The tasks use directories because each one owns different instructions and verifier logic. Its `task.yaml` keeps those
references beside its instruction, verifier, fixtures, repositories, tests, or other assets. The root manifest points
to the collection rather than enumerating every task.

Use JSONL instead when many tasks share the same structure and differ only in data values. [Hello Taskset](../hello_taskset/README.md) demonstrates that data-driven form.

## Related examples

- [Hello World](../hello_world/README.md) — Create the smallest single-task environment.
- [Hello Taskset](../hello_taskset/README.md) — Run a named collection of similar tasks.
- [Hello MCP Tool](../hello_mcp_tool/README.md) — Give an agent an additional tool over MCP.
