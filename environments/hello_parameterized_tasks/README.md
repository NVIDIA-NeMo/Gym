# Hello Parameterized Tasks

Hello Parameterized Tasks helps you add and update many similarly structured tasks without copying their common setup or scoring logic.

## Concept

Parameterized tasks keep the common task files in one place and store the values that vary in one JSONL record per task. NeMo Gym validates the records and creates a self-contained task folder for each one before execution. The generated tasks form a dataset that can be selected and run together. Values used only for scoring do not need to appear in the agent's instruction.

For definitions of task, software environment, sandbox, verifier, and reward, see [Terms used in these examples](../hello_world/README.md#terms-used-in-these-examples).

### Examples suited to JSONL

JSONL works well when one shared task definition can turn every row into an instruction and score the result. Common examples include:

- Math problems with a question and expected answer
- Q&A tasks, including multiple-choice questions, with their reference answers
- Instruction-following tasks with an instruction and scoring criteria

JSONL is also convenient when tasks are imported or generated in bulk and need to be filtered, split, or sharded.

### When to use parameterized tasks

Use JSONL when every task can share the same environment, instruction pattern, and verifier implementation. Each row supplies the values that change from task to task.

Use separate task folders when individual tasks need their own files, dependencies, setup, tools, or scoring logic.

## In this example

- `tasks.jsonl` contains one question and expected answer per line.
- `task_data.py` is the `TaskData` model every row must satisfy (`question` and `expected_answer`, both strings,
  nothing else); a row that does not is refused at prepare with its task named, and again at seed and collate.
- `dataset.toml` names the dataset and its version, and holds NeMo Gym settings in its `[gym]` table.
- `instruction.template.md` inserts each question into the request shown to the agent.
- `task.toml`, `environment/`, and `tests/` are shared by every row.
- `tests/verifier.py` receives the expected answer without exposing it to the agent, and reads the agent's
  `/workspace/answer.md`.
- `solution/answer.template.md` is the reference answer, rendered per row into `solution/answer.md`.

The generated task folders can be inspected, exported, and run like any other.

## Run it

```bash
gym eval run environments/hello_parameterized_tasks --agent opencode_agent --sandbox docker \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The command uses OpenCode because it can work with files and tools in the task sandbox while keeping model choice separate. It uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

## Validate it

Every task ships a `solution/`: here the reference answer, `solution/answer.md`, rendered per row from
`answer.template.md`. The oracle writes it to `answer.md` in the working directory (the file the verifier reads),
runs the verifier and expects reward 1:

```bash
gym dataset validate environments/hello_parameterized_tasks --sandbox docker
```

On a cluster, pass `--sandbox opensandbox` instead.

## Check the result

`gym dataset validate` ends with one line per task:

```text
task_id	status	reward	verify_s	network
addition [repeat 0]	oracle solved	1.0	0.8	public
multiplication [repeat 0]	oracle solved	1.0	0.8	public
```

Each task passed when its line reads `oracle solved` and reward `1.0`; the command then exits 0.

Each command writes its results to its own folder: `results/harbor/validation/hello_parameterized_tasks/` for validate and `results/harbor/hello_parameterized_tasks/` for the agent run. In either folder, `rollouts_run_summary.json` lists each task's `status` and `reward`, and the paths of its verifier logs, where `reward.txt` and `test-stdout.txt` show the reward `verify()` returned for each row and the verifier's output.

## Other examples

### Tasks and datasets

- [Hello World](../hello_world/README.md): Define one self-contained task folder.
- [Hello Dataset](../hello_dataset/README.md): Run multiple distinct tasks as one dataset.

### Add tools

- [Hello MCP Tool](../hello_mcp_tool/README.md): Give an agent an additional tool over MCP.

### Python scoring

- [Hello Python Verifier](../hello_python_verifier/README.md): Write task-specific scoring logic in Python.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.
