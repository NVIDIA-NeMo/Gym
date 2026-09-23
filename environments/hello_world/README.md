# Hello World

Hello World gives you a minimal starting point for creating your first NeMo Gym task.

## Concept

A directory task keeps the instruction, execution environment, scoring logic, and resource requirements together, making the complete task easy to inspect, share, and run. NeMo Gym adopts Harbor's task format for directory tasks, so compatible Harbor tasks can run directly in NeMo Gym.

## In this example

- `instruction.md` asks the agent to create one text file.
- `environment/Dockerfile` defines the software environment where the agent works.
- `tests/test.sh` checks the file and returns `1` when its contents are correct.
- `task.toml` identifies the task and declares its resource requirements.

These files form one complete task directory.

## Run it

```bash
export OPENAI_API_KEY="<api-key>"

gym eval run environments/hello_world \
  --agent opencode \
  --model openai/gpt-5 \
  --sandbox docker
```

The command uses OpenCode because it can work with files and tools in the task sandbox while keeping model choice separate. It uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

## Check the result

TODO: After runtime integration is available, add the literal successful CLI output and identify the exact result fields and artifacts a user should inspect to verify the score.

## Other examples

### Tasks and datasets

- [Hello Dataset](../hello_dataset/README.md): Run multiple distinct tasks as one dataset.

### Add tools

- [Hello MCP Tool](../hello_mcp_tool/README.md): Give an agent an additional tool over MCP.

### Python scoring

- [Hello Python Verifier](../hello_python_verifier/README.md): Write task-specific scoring logic in Python.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
