# Hello World

Hello World gives you a minimal starting point for creating your first NeMo Gym task.

## Concept

A directory task keeps the instruction, execution environment, scoring logic, and resource requirements together, making the complete task easy to inspect, share, and run. NeMo Gym adopts Harbor's task format for directory tasks, so compatible Harbor tasks can run directly in NeMo Gym.

### Terms used in these examples

- **Task:** One complete unit of work, including its instruction, software environment, resource requirements, and verification.
- **Instruction:** What the agent is asked to do.
- **Software environment:** The tools and dependencies available to the agent. For this task, `environment/Dockerfile` defines that environment.
- **Sandbox:** An isolated running instance of the software environment, created by a provider such as Docker.
- **Verifier:** The code that scores the agent's work.
- **Reward:** The numeric result returned by the verifier.

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

### What NeMo Gym starts

NeMo Gym separates an evaluation into services so the model, agent harness, task state and scoring, and rollout workflow can be changed or scaled independently.

For this tutorial and the common task-authoring workflow, you can treat these services as implementation details. You define the task and select a compatible agent, model, and sandbox; the CLI composes the services for you automatically. You only need to work with individual services for custom runtime behavior or advanced deployments. Here's what's happening behind the scenes:

- The **Environment Server** coordinates setup, agent execution, verification, and cleanup for each rollout.
- The **Agent Server** runs the agent harness.
- The **Model Server** provides inference.
- The **Resources Server** makes task-defined tools, state, and verification capabilities available at runtime.

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
