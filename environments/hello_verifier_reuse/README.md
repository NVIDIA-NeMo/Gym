# Hello Verifier Reuse

Hello Verifier Reuse helps you keep scoring consistent across tasks without copying and maintaining the same Python logic.

## Concept

Python verifiers can reuse a complete verifier or build task-specific checks from shared helpers. The shared code can come from NeMo Gym's curated collection of reusable verifiers and helper functions, another Python package, or code maintained with the dataset.

NeMo Gym packages the referenced code with each materialized task so the resulting task remains reproducible.

Prototype dependency: This example uses the proposed `nemo_gym.verifiers.files` API. That core module must be added and tested in the separate runtime implementation before this example can run.

## In this example

This example is one dataset with two complete task directories at its root:

```text
hello_verifier_reuse/
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
export OPENAI_API_KEY="<api-key>"

gym eval run environments/hello_verifier_reuse \
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

### Parameterized tasks

- [Hello Parameterized Tasks](../hello_parameterized_tasks/README.md): Generate many tasks from shared files and JSONL records.
