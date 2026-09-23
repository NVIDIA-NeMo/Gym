# Hello Dataset

Hello Dataset helps you organize related tasks so they can be selected and run together while each task keeps its own requirements.

## Concept

Harbor calls a runnable collection of tasks a dataset. In a local dataset, each immediate subdirectory is one complete task. Tasks can belong to the same dataset without sharing instructions, environments, assets, or scoring logic.

## In this example

- `hello/` asks for one greeting file.
- `goodbye/` asks for a different file and checks different contents.

The two tasks are independent. Open either directory and you will find its instruction, environment, task configuration, and verifier.

## Run it

```bash
export OPENAI_API_KEY="<api-key>"

gym eval run environments/hello_dataset \
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

### Add tools

- [Hello MCP Tool](../hello_mcp_tool/README.md): Give an agent an additional tool over MCP.

### Python scoring

- [Hello Python Verifier](../hello_python_verifier/README.md): Write task-specific scoring logic in Python.
- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
