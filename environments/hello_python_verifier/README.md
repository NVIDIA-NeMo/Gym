# Hello Python Verifier

Hello Python Verifier lets you score an agent's work with familiar Python code and libraries instead of translating the checks into shell commands.

## Concept

Python is useful when scoring relies on existing validation code, structured data, or domain libraries that would be awkward to use from a shell script. A task defines one scoring entry point with either `tests/test.sh` or `tests/verifier.py`. For a Python verifier, NeMo Gym calls the file's `verify()` function and uses the number it returns as the reward. NeMo Gym handles the shell launcher and reward files.

## In this example

`tests/verifier.py` checks whether the agent created `hello-gym.txt` with the expected contents. The verifier returns `1.0` when the file matches and `0.0` when it is missing or incorrect.

## Run it

```bash
export OPENAI_API_KEY="<api-key>"

gym eval run environments/hello_python_verifier \
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

- [Hello Verifier Reuse](../hello_verifier_reuse/README.md): Reuse Python verifiers and shared helper functions.

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
