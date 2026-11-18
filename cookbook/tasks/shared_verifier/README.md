# Shared Verifier

This recipe helps you keep scoring consistent across tasks without copying and maintaining the same Python logic.

## Concept

Python verifiers can reuse a complete verifier or build task-specific checks from shared helpers. The shared code can come from NeMo Gym's curated collection of reusable verifiers and helper functions, another Python package, or code maintained with the dataset.

NeMo Gym packages the referenced code with each materialized task so the resulting task remains reproducible.

The shared helpers come from `nemo_gym.verifiers.files`, which NeMo Gym vendors into the verifier sandbox beside the task's `verifier.py`, so nothing needs installing in the task image.

For definitions of task, software environment, sandbox, verifier, and reward, see [Terms used in these examples](../hello_world/README.md#terms-used-in-these-examples).

## In this example

This example is one dataset with two complete task folders at its root:

```text
shared_verifier/
├── exact-greeting/
│   ├── instruction.md
│   ├── task.toml
│   ├── environment/Dockerfile
│   └── tests/verifier.py
└── uppercase-greeting/
    ├── instruction.md
    ├── task.toml
    ├── environment/Dockerfile
    └── tests/verifier.py
```

Both tasks need to inspect a text file, so their `tests/verifier.py` entry points reuse tested NeMo Gym code instead of duplicating common checks. Each task only defines what is unique:

- `exact-greeting/tests/verifier.py` reuses NeMo Gym's complete `text_file_equals` verifier with task-specific settings.
- `uppercase-greeting/tests/verifier.py` defines its own verifier and uses NeMo Gym's `read_text` helper inside it.

Both files expose the same `verify()` entry point. The difference is how much scoring logic each task reuses.

## Run it

```bash
gym eval run cookbook/tasks/shared_verifier --agent opencode_agent --sandbox docker \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The command uses OpenCode because it can work with files and tools in the task sandbox while keeping model choice separate. It uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

## Validate it

Every task ships a `solution/` (a `solve.sh` here; the parameterized dataset ships a `solution/answer.md` rendered per
row instead). The oracle plays it through the task's verifier and expects reward 1:

```bash
gym dataset validate cookbook/tasks/shared_verifier --sandbox docker
```

On a cluster, pass `--sandbox opensandbox` instead.

## Check the result

`gym dataset validate` ends with one line per task:

```text
task_id	status	reward	verify_s	network
exact-greeting [repeat 0]	oracle solved	1.0	0.9	public
uppercase-greeting [repeat 0]	oracle solved	1.0	0.9	public
```

Each task passed when its line reads `oracle solved` and reward `1.0`; the command then exits 0.

Each command writes its results to its own folder: `results/harbor/validation/shared_verifier/` for validate and `results/harbor/shared_verifier/` for the agent run. In either folder, `rollouts_run_summary.json` lists each task's `status` and `reward`, and the paths of its verifier logs, where `reward.txt` and `test-stdout.txt` show the reward each task's `verify()` returned and the verifier's output.

## Other examples

### Tasks and datasets

- [Hello World](../hello_world/README.md): Write your first task, run an agent on it, and see its score.
- [Dataset](../dataset/README.md): Run several different tasks together.

### Add tools

- [MCP Tool](../mcp_tool/README.md): Give the agent a tool it needs for the task.

### Python scoring

- [Python Verifier](../python_verifier/README.md): Check the agent's work with Python.

### Parameterized tasks

- [Parameterized Tasks](../parameterized_tasks/README.md): Make many versions of one task that differ only in their inputs.
