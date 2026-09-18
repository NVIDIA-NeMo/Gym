# Sandboxed Hermes

`HermesHarness` runs the pinned native Hermes agent with model calls routed
through Gym and task commands executed in a caller-owned `AsyncSandbox`.
Like `miniswe_sandboxed_agent`, it uses the resources runner to provision the
sandbox, stop task processes, verify the result, and clean up.

`app.py` forwards Gym `/run` requests to the resources runner, including rollout
identity and model-capture settings. `harness.py` accepts the sandbox, task
context, model callback, and artifact directory. Hermes and mini-SWE share
`HarnessContext` and `HarnessOutcome` from `nemo_gym.sandbox.harness`.

Hermes runs in a separate worker process for each episode because its native
terminal backend registry and interruption state are process globals.
`worker.py` supplies model and sandbox adapters to the native agent. Requests
cross a local pipe to the async Gym callbacks; native tool dispatch, model
retries, and parallel tool execution remain in Hermes. Cancellation stops
pending callbacks and joins the worker before returning to the resources runner.

`configs/hermes.yaml` selects native terminal and file toolsets: `terminal`,
`process`, `read_file`, `write_file`, `patch`, and `search_files`. Their schemas
and handlers come from Hermes, including background-process support. The YAML
also supplies native agent settings and runtime overrides. Each episode writes
those overrides under its own artifact directory in `home/config.yaml`.
Hermes supplies the remaining defaults; the harness does not set a model
context length. Compression and memory are disabled for this profile. Task MCP
servers are rejected during setup; tasks can provide a skills directory.

The benchmark profile is `benchmarks/terminal_bench_4/hermes.yaml`. It uses the
existing TB4 resources runner and pinned task packages. Defaults are 90 native
agent iterations and 30 seconds per sandbox execution request; `tb4_max_steps` and
`tb4_step_timeout_sec` override them. Background processes can outlive an individual
request. Hermes may make a final summary model call
when it reaches the iteration limit. The task's overall deadline still applies.

Model calls use Gym's Responses API callback. The returned trajectory retains
accepted model output, reasoning, matching tool calls and results, and reported
token usage. Native messages are saved to `trajectory.json`; worker diagnostics
are saved to `worker.log`. The harness also returns Gym's structured trajectory
with model turns and tool callback timings. For parallel batches, callback
durations can include time waiting for another tool. Enable `observability_enabled` and set an
absolute `model_call_capture_dir` to join turns to captured model requests and
responses in rollout artifacts.

The Hermes dependency is pinned to `26bb847a88493342ca1b194e0455b479073ae21d`.
Run the agent tests with:

```bash
uv run gym env test +entrypoint=responses_api_agents/hermes_sandboxed_agent
```
