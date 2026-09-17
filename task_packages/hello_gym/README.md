# Hello Gym

Hello Gym asks an agent to create a text file with a specific message. Gym gives the agent a fresh workspace, records what it does, and verifies the resulting file.

Define the tasks, environment, and scoring once. Then compare agents and
models without rewriting any of them.

## Tasks

The package includes two small tasks:

- **Hello** creates `/workspace/hello-gym.txt` with
  `Hello from NeMo Gym!`.
- **Goodbye** creates `/workspace/goodbye-gym.txt` with
  `Goodbye from NeMo Gym!`.

Both use the same instruction template, environment, and verifier. Only the
target path and expected content differ.

## Run it

Set an API key, then run the example:

```bash
export ANTHROPIC_API_KEY="your-api-key"

gym eval run \
  --task-package hello-gym \
  --taskset example \
  --agent claude_code_agent \
  --model claude-sonnet-4-6
```

Gym loads the package, renders each task's instruction, creates a fresh
workspace, runs the selected agent and model, invokes the package verifier,
and reports the reward for each attempt. Change `--agent` or `--model` to run
the same tasks with another compatible pairing while keeping the environment
and scoring rules fixed.

> [!NOTE]
> This is the target interface. TaskPackage loading and execution are not
> implemented yet.

## How it works

For each task, Gym:

1. Creates a clean workspace from `environment/Dockerfile`.
2. Gives the instruction to the agent.
3. Lets the agent work in `/workspace`.
4. Records the agent's response and actions.
5. Checks that the requested file exists with the expected content.
6. Returns reward `1.0` for success or `0.0` otherwise.

Hello Gym renders both tasks from one `instruction.md` template, runs them in
the same environment, and checks them with one `verifier.py`. Each task
supplies only the path and content that differ. Keeping those shared
definitions together makes results reproducible across compatible agents and
models.

## Future work

Companion examples can introduce one additional concept at a time:

- Reuse a Gym-provided verifier instead of a package-local verifier.
- Declare an agent-facing MCP server and use it during a task.
