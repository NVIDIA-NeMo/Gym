# NeMo Gym Cookbook

Small, runnable recipes for common jobs with NeMo Gym. Every recipe here is tested in CI and ships in the `nemo-gym` package.

## Start here

[Hello World](tasks/hello_world/README.md): write your first task, run an agent on it, and see its score. It ends with links to the next step for each goal below.

## Recipes by goal

### Build tasks and verifiers

- [Hello World](tasks/hello_world/README.md): write your first task, run an agent on it, and see its score.
- [Dataset](tasks/dataset/README.md): run several different tasks together.
- [Parameterized Tasks](tasks/parameterized_tasks/README.md): make many versions of one task that differ only in their inputs.
- [Python Verifier](tasks/python_verifier/README.md): check the agent's work with Python.
- [Shared Verifier](tasks/shared_verifier/README.md): use the same checks in many tasks.
- [MCP Tool](tasks/mcp_tool/README.md): give the agent a tool it needs for the task.

### Run an evaluation

No recipes yet.

### Analyze failures and compare results

No recipes yet.

### Optimize an agent

No recipes yet.

### Train a model

No recipes yet.

## Add a recipe

Copy the closest recipe to `cookbook/<group>/<name>/`, list it above, and run `pytest tests/unit_tests/test_cookbook.py`, which checks what every recipe needs.
