# Hello MCP Tool

Hello MCP Tool lets you give an agent a tool it needs for a task without changing the agent itself.

## Concept

A task can declare required tools through the Model Context Protocol (MCP). NeMo Gym refuses each episode to an agent that cannot provide those tools, rather than run it without them. MCP gives tasks a standard way to describe and connect them.

For definitions of task, software environment, sandbox, verifier, and reward, see [Terms used in these examples](../hello_world/README.md#terms-used-in-these-examples).

## In this example

- `task.toml` declares the `hello-tools` MCP server as a task requirement.
- `environment/tools/server.py` implements a `get_greeting` tool.
- `environment/Dockerfile` installs the server and its dependencies in the task environment.
- `instruction.md` asks the agent to use the tool and save its result.
- `tests/test.sh` checks the saved result.

## Run it

```bash
gym eval run environments/hello_mcp_tool --agent miniswe_sandboxed_agent --sandbox docker \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The command uses mini-SWE because it starts the task's `stdio` MCP server inside the sandbox. A harness that does not, such as OpenCode, is refused this task. It uses Docker for a simple local sandbox, which you can replace with another sandbox provider.

## Validate it

Every task ships a `solution/` (a `solve.sh` here; the parameterized dataset ships a `solution/answer.md` rendered per
row instead). The oracle plays it through the task's verifier and expects reward 1:

```bash
gym dataset validate environments/hello_mcp_tool --sandbox docker
```

On a cluster, pass `--sandbox opensandbox` instead, and set `builds.repository` to a registry the sandboxes can pull
from, because this task's Dockerfile needs a build (see "Harbor image builds" in
`fern/versions/latest/pages/reference/configuration.mdx`).

## Check the result

`gym dataset validate` ends with one line per task:

```text
task_id	status	reward	verify_s	network
hello_mcp_tool [repeat 0]	oracle solved	1.0	1.2	public
```

The task passed when its line reads `oracle solved` and reward `1.0`; the command then exits 0.

Each command writes its results to its own folder: `results/harbor/validation/hello_mcp_tool/` for validate and `results/harbor/hello_mcp_tool/` for the agent run. In either folder, `rollouts_run_summary.json` lists each task's `status` and `reward`, and the paths of its verifier logs, where `reward.txt` and `test-stdout.txt` show what `tests/test.sh` wrote.

## Other examples

### Tasks and datasets

- [Hello World](../hello_world/README.md): Define one self-contained task folder.
- [Hello Dataset](../hello_dataset/README.md): Run multiple distinct tasks as one dataset.

### Python scoring

- [Hello Python Verifier](../hello_python_verifier/README.md): Write task-specific scoring logic in Python.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
