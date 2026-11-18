# Hello World

Hello World gives you a minimal starting point for creating your first NeMo Gym task.

Install NeMo Gym first: see [Installation](https://docs.nvidia.com/nemo/gym/main/get-started/installation).

## Concept

A task folder keeps the instruction, execution environment, scoring logic, and resource requirements together, making the complete task easy to inspect, share, and run. Task folders written for Harbor also run in NeMo Gym as they are.

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

These files form one complete task folder.

## Run it

```bash
gym eval run cookbook/tasks/hello_world --agent opencode_agent --sandbox docker \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The command uses OpenCode (`opencode_agent`; the short name `opencode` matches two harnesses and is refused) because it can work with files and tools in the task sandbox while keeping model choice separate. OpenCode runs inside the sandbox and calls Gym's model server on the host, which calls `<url>`, the base URL of any OpenAI-compatible endpoint. So the sandbox must reach the Gym host, and the host must reach `<url>`. The command uses Docker for a simple local sandbox. You can replace either one with another compatible agent or sandbox provider.

### What NeMo Gym starts

NeMo Gym separates an evaluation into services so the model, agent harness, task state and scoring, and rollout workflow can be changed or scaled independently.

For this tutorial and the common task-authoring workflow, you can treat these services as implementation details. You define the task and select a compatible agent, model, and sandbox; the CLI composes the services for you automatically. You only need to work with individual services for custom runtime behavior or advanced deployments. Here's what's happening behind the scenes:

- The **Environment Server** coordinates setup, agent execution, verification, and cleanup for each rollout.
- The **Agent Server** runs the agent harness.
- The **Model Server** provides inference.
- The task's own `tests/test.sh` does the scoring, inside the sandbox. No Resources Server takes part.

## Validate it

Every task ships a `solution/` (a `solve.sh` here; the parameterized dataset ships a `solution/answer.md` rendered per
row instead). The oracle plays it through the task's verifier and expects reward 1:

```bash
gym dataset validate cookbook/tasks/hello_world --sandbox docker
```

On a cluster, pass `--sandbox opensandbox` instead.

## Check the result

`gym dataset validate` ends with one line per task. The task passed when its line reads `hello_world [repeat 0]`, `oracle solved` and reward `1.0`; the command then exits 0.

Each command writes its results to its own folder: `results/harbor/validation/hello_world/` for validate and `results/harbor/hello_world/` for the agent run. In either folder, `rollouts_run_summary.json` lists each task's `status` and `reward`, and the paths of its verifier logs, where `reward.txt` and `test-stdout.txt` show what `tests/test.sh` wrote.

## Next steps

Pick the goal that fits what you want to do next. Each links to its section of the [cookbook index](../../README.md).

- Build more tasks and verifiers:
  - [Dataset](../dataset/README.md): run several different tasks together.
  - [MCP Tool](../mcp_tool/README.md): give the agent a tool it needs for the task.
  - [Python Verifier](../python_verifier/README.md): check the agent's work with Python.
  - [Shared Verifier](../shared_verifier/README.md): use the same checks in many tasks.
  - [Parameterized Tasks](../parameterized_tasks/README.md): make many versions of one task that differ only in their inputs.
- [Run an evaluation](../../README.md#run-an-evaluation) of an agent or model on existing tasks.
- [Analyze failures and compare results](../../README.md#analyze-failures-and-compare-results), such as why an agent failed a task or how two versions of your agent differ.
- [Optimize an agent](../../README.md#optimize-an-agent) without training.
- [Train a model](../../README.md#train-a-model).
