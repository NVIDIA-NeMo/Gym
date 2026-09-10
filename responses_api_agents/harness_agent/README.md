# harness_agent

Runs a configured agent harness through a sandbox provider. LocalProvider runs the same
path without container isolation.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agnostic, not the agent config. 
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | in-box dir the agent's `repo_dir` points at, so edits land in the graded tree |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward lands in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |
Tasks with an external verifier (e.g. math) need none of these beyond an image.

## In-sandbox agent runtime

The server tars `nemo_gym/` and `responses_api_agents/` at startup (small, data and tests
excluded) and unpacks it to `/gym_mount` in each sandbox. `setup_commands` install the
agent's dependencies, for example `pip install nemo-gym` for the import chain plus the
harness CLI itself.

Because the harness inside the sandbox talks to a standard Gym model server, this agent
composes with future model-server capabilities (e.g. token-ID capture for training)
without changes to the harness or this server.
