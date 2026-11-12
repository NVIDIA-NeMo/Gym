# Hello Python Verifier

Hello Python Verifier lets you score an agent's work with familiar Python code and libraries instead of translating the checks into shell commands.

## Concept

Python is useful when scoring relies on existing validation code, structured data, or domain libraries that would be awkward to use from a shell script. A task defines one scoring entry point with either `tests/test.sh` or `tests/verifier.py`. For a Python verifier, NeMo Gym calls the file's `verify()` function and uses the number it returns as the reward. NeMo Gym handles the shell launcher and reward files: a `tests/test.sh` shim runs `verify()` inside the sandbox with the working directory as `workspace`.

For definitions of task, software environment, sandbox, verifier, and reward, see [Terms used in these examples](../hello_world/README.md#terms-used-in-these-examples).

## In this example

`tests/verifier.py` checks whether the agent created `hello-gym.txt` with the expected contents. The verifier returns `1.0` when the file matches and `0.0` when it is missing or incorrect.

## Run it

```bash
gym eval run environments/hello_python_verifier --agent opencode_agent --sandbox docker \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The command uses OpenCode because it can work with files and tools in the task sandbox while keeping model choice separate. It uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

## Validate it

Every task ships a `solution/` (a `solve.sh` here; the parameterized dataset ships a `solution/answer.md` rendered per
row instead). The oracle plays it through the task's verifier and expects reward 1:

```bash
gym dataset validate environments/hello_python_verifier --sandbox docker
```

On a cluster, pass `--sandbox opensandbox` instead.

## Check the result

`gym dataset validate` ends with one line per task:

```text
task_id	status	reward	verify_s	network
hello_python_verifier [repeat 0]	oracle solved	1.0	0.9	public
```

The task passed when its line reads `oracle solved` and reward `1.0`; the command then exits 0.

Each command writes its results to its own folder: `results/harbor/validation/hello_python_verifier/` for validate and `results/harbor/hello_python_verifier/` for the agent run. In either folder, `rollouts_run_summary.json` lists each task's `status` and `reward`, and the paths of its verifier logs, where `reward.txt` and `test-stdout.txt` show the reward `verify()` returned and the verifier's output.

## Other examples

### Tasks and datasets

- [Hello World](../hello_world/README.md): Define one self-contained task folder.
- [Hello Dataset](../hello_dataset/README.md): Run multiple distinct tasks as one dataset.

### Add tools

- [Hello MCP Tool](../hello_mcp_tool/README.md): Give an agent an additional tool over MCP.

### Python scoring

- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
