# Hello Parameterized Tasks

Hello Parameterized Tasks helps you add and update many similarly structured tasks without copying their common setup or scoring logic.

## Concept

Parameterized tasks keep the common task files in one place and store the values that vary in one JSONL record per task. NeMo Gym validates the records and creates a self-contained task directory for each one before execution. The generated tasks form a dataset that can be selected and run together. Values used only for scoring do not need to appear in the agent's instruction.

For definitions of task, software environment, sandbox, verifier, and reward, see [Terms used in these examples](../hello_world/README.md#terms-used-in-these-examples).

### Examples suited to JSONL

JSONL works well when one shared task definition can turn every row into an instruction and score the result. Common examples include:

- Math problems with a question and expected answer
- Q&A tasks, including multiple-choice questions, with their reference answers
- Instruction-following tasks with an instruction and scoring criteria

JSONL is also convenient when tasks are imported or generated in bulk and need to be filtered, split, or sharded.

### When to use parameterized tasks

Use JSONL when every task can share the same environment, instruction pattern, and verifier implementation. Each row supplies the values that change from task to task.

Use separate task directories when individual tasks need their own files, dependencies, setup, tools, or scoring logic.

## In this example

- `tasks.jsonl` contains one question and expected answer per line.
- `instruction.template.md` inserts each question into the request shown to the agent.
- `task.toml`, `environment/`, and `tests/` are shared by every row.
- `tests/verifier.py` receives the expected answer without exposing it to the agent.

The generated task directories can be inspected, exported, and run as an ordinary dataset in NeMo Gym or Harbor.

## Run it

```bash
export OPENAI_API_KEY="<api-key>"

gym eval run environments/hello_parameterized_tasks \
  --agent opencode \
  --model openai/gpt-5 \
  --sandbox docker
```

The command uses OpenCode because it can work with files and tools in the task sandbox while keeping model choice separate. It uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

## Check the result

TODO: After runtime integration is available, add the literal successful CLI output and identify the exact result fields and artifacts a user should inspect to verify the score.

## Other examples

### Tasks and datasets

- [Hello World](../hello_world/README.md): Define one self-contained directory task.
- [Hello Dataset](../hello_dataset/README.md): Run multiple distinct tasks as one dataset.

### Add tools

- [Hello MCP Tool](../hello_mcp_tool/README.md): Give an agent an additional tool over MCP.

### Python scoring

- [Hello Python Verifier](../hello_python_verifier/README.md): Write task-specific scoring logic in Python.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.
