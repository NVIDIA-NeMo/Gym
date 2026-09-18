# Sandboxed Hermes

`HermesHarness` runs the pinned Hermes conversation loop with a terminal tool
backed by a caller-owned `AsyncSandbox`. Like `miniswe_sandboxed_agent`, the
Python agent loop runs in the resources process; task commands run in the
sandbox. The resources runner provisions the sandbox, stops task processes,
verifies the result, and cleans up.

`app.py` forwards Gym `/run` requests to that resources runner, including the
rollout identity and model-capture settings. `harness.py` accepts the sandbox,
task instruction, working directory, user, model callback, and artifact
directory. It contains no benchmark loading or verification logic.

The profile is `benchmarks/terminal_bench_4/hermes.yaml`. It uses the existing
TB4 resources runner and pinned task packages. Defaults are 90 model turns
and 30 seconds per command; `tb4_max_steps` and `tb4_step_timeout_sec` override
them. The task's overall deadline still applies.

Model calls go through Gym's Responses API callback. The returned trajectory
contains accepted model output, reasoning, matching tool calls and results, and reported
token usage. Native Hermes messages are saved to `trajectory.json` after each
model request and tool result. Timeout and cancellation cancel pending I/O and
join the Hermes worker before control returns to the resources runner.

This terminal profile exposes one foreground bash command per tool call.
Each command starts in the task working directory. Hermes context compression,
memory, delegation, and additional toolsets are disabled. Hermes uses an isolated
process configuration with a 128,000-token context estimate. Task MCP servers are
rejected during setup; tasks can provide skills through their skills directory.
The Hermes dependency is pinned to `26bb847a88493342ca1b194e0455b479073ae21d`.

Run the agent tests with:

```bash
uv run gym env test +entrypoint=responses_api_agents/hermes_sandboxed_agent
```
