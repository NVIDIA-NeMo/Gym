# Hello Dataset

Hello Dataset helps you organize related tasks so they can be selected and run together while each task keeps its own requirements.

## Concept

A dataset is a folder of tasks that run together. Each immediate subfolder is one complete task. Tasks can belong to the same dataset without sharing instructions, environments, assets, or scoring logic.

For definitions of task, software environment, sandbox, verifier, and reward, see [Terms used in these examples](../hello_world/README.md#terms-used-in-these-examples).

## In this example

- `hello/` asks for one greeting file.
- `goodbye/` asks for a different file and checks different contents.

The two tasks are independent. Open either directory and you will find its instruction, environment, task configuration, and verifier.

## Run it

```bash
gym eval run environments/hello_dataset --agent opencode_agent --sandbox docker \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The command uses OpenCode because it can work with files and tools in the task sandbox while keeping model choice separate. It uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

## Validate it

Every task ships a `solution/` (a `solve.sh` here; the parameterized dataset ships a `solution/answer.md` rendered per
row instead). The oracle plays it through the task's verifier and expects reward 1:

```bash
gym dataset validate environments/hello_dataset --sandbox docker
```

On a cluster, pass `--sandbox opensandbox` instead.

## Check the result

`gym dataset validate` ends with one line per task:

```text
task_id	status	reward	verify_s	network
goodbye [repeat 0]	oracle solved	1.0	0.4	public
hello [repeat 0]	oracle solved	1.0	0.5	public
```

Each task passed when its line reads `oracle solved` and reward `1.0`; the command then exits 0.

Each command writes its results to its own folder: `results/harbor/validation/hello_dataset/` for validate and `results/harbor/hello_dataset/` for the agent run. In either folder, `rollouts_run_summary.json` lists each task's `status` and `reward`, and the paths of its verifier logs, where `reward.txt` and `test-stdout.txt` show what each task's `tests/test.sh` wrote.

## Other examples

### Tasks and datasets

- [Hello World](../hello_world/README.md): Define one self-contained task folder.

### Add tools

- [Hello MCP Tool](../hello_mcp_tool/README.md): Give an agent an additional tool over MCP.

### Python scoring

- [Hello Python Verifier](../hello_python_verifier/README.md): Write task-specific scoring logic in Python.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
