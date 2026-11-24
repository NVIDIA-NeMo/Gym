# harness_agent

> [!WARNING]
> **Deprecated.** `harness_agent` runs each episode through its own legacy `/run` path, behind a `legacy_agent`
> environment server, and will be removed after the environment-server migration. Do not build new benchmarks,
> agents or configs on it. Use the harness's own agent with sandbox sessions under an environment server instead
> (`claude_code_agent`, `codex_agent`, `hermes_agent`, `pi_agent`, `opencode_agent` and so on); the `job_bench` and
> `workspace_bench` configs do this through `single_agent_turn_legacy`. Sandbox handling stays shared on that path:
> the resources server starts, lends and stops the task sandbox (`SandboxSessionResourcesServer`), the environment
> server orders seed, turn, verify and close, and each agent launches and stops its CLI in the lent sandbox through
> the shared helpers in `nemo_gym/agents/utils/`.
>
> The agent logs a deprecation warning at startup. Gym no longer ships a preset for it.

Runs any gym agent harness in a sandbox with any resources server.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agnostic, not the agent config. 
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | in-box dir the agent's `repo_dir` points at, so edits land in the graded tree, unless `agent_kwargs` sets that key |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward goes in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |


Tasks with an external verifier (e.g. math) need none of these beyond an image.

## Model endpoint

The harness inside the sandbox calls the Gym model server named by `model_server` (its `host:port`), so the
server's request handling, model-call capture and training token capture apply to every in-box call.

Set `sandbox_model_base_url` only when the sandbox cannot reach that `host:port`, for example sandboxes on a
cluster without a route to compute nodes. Point it at a relay, tunnel or gateway that forwards to the same
model server. A URL that bypasses the model server (such as a backend vLLM or API URL) skips its handling and
capture.
