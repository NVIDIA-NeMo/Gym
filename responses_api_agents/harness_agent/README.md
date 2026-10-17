# harness_agent

Runs any gym agent harness in a sandbox with any resources server.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agnostic, not the agent config. 
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | in-box dir the agent's `repo_dir` points at, so edits land in the graded tree |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward goes in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |


Tasks with an external verifier (e.g. math) need none of these beyond an image.

## Model address used inside the sandbox

Each rollout selects one model URL and uses it for every turn. The agent logs the final URL at INFO level,
including the rollout/capture path prefix when present. Check this log to confirm where model calls go.

For training, set `sandbox_model_base_url` to the sandbox-reachable **Gym model server** address,
for example `http://<head-ip>:<gym-model-port>`. Calls then pass through the model server's configured
message rewriting and sampling overrides. With `vllm_model`, routing is stable per rollout when its
prefix is present (otherwise per session). Health-based reassignment requires
`route_around_failing_endpoints: true`; it is off by default.

Passing through Gym is also required for its training token capture, but does not enable capture by
itself. Configure run-level `token_id_capture.enabled` and the agent's `token_id_capture`
flag (or run-level `token_id_capture.all_agents`) so requests carry `/ng-rollout/<id>/training-token-capture`.
The backend must support the configured token capture mode. Verify capture records before training.

When `sandbox_model_base_url` is unset, the agent uses the referenced model config's `base_url` if
present; otherwise it uses that Gym server's `host:port`. A backend `base_url` therefore sends calls
directly to that backend, bypassing Gym's model-server processing and capture. A config without
`base_url` falls back to the Gym server. Inferred loopback addresses are rewritten for sandbox access;
explicit overrides, including loopback addresses, are used unchanged apart from the trailing `/v1`
and slash normalization and the appended rollout/capture prefix.

Both an explicit `sandbox_model_base_url` list and an inferred backend `base_url` list are assigned
round-robin per rollout, within each agent-server process. Previously, an inferred list always selected
its first endpoint. Every turn of a rollout keeps its selected URL. This selection has no health checks
or failover: a dead entry still receives one of every N rollout starts. Remove dead entries yourself,
or point to a Gym model server and enable its health routing. Direct backends must accept any forwarded
rollout/capture path prefix.
