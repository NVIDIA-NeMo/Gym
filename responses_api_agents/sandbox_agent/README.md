# sandbox_agent

Runs any Gym environment inside a sandbox.

## Modes

- agent_only_runner: imports the configured agent's `responses()` inside the sandbox via
  a small runner. The harness code executes unchanged, only its model base URL is patched
  to a sandbox-reachable address. No Gym servers in the sandbox.
- gym_runner: starts full NeMo Gym inside the sandbox (`nested_config_paths`) and posts
  the task to its agent `/run` endpoint, unwrapping the verify payload so the reward
  lands in response metadata as `sandbox_reward`. For wrapping environments without a
  clean responses/verify split.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agnostic, not the agent config. 
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | in-box dir the agent's `repo_dir` points at, so edits land in the graded tree |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward lands in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |
| `patch_workdir` | in-box dir whose `git diff` is captured into response metadata as `model_patch` for hermetic grading by a resources server |

Tasks with an external verifier (e.g. math) need none of these beyond an image.

## In-sandbox agent runtime

The server tars `nemo_gym/` and `responses_api_agents/` at startup (small, data and tests
excluded) and unpacks it to `/gym_mount` in each sandbox. `setup_commands` install the
agent's dependencies, for example `pip install nemo-gym` for the import chain plus the
harness CLI itself.

Because the harness inside the sandbox talks to a standard Gym model server, this agent
composes with future model-server capabilities (e.g. token-ID capture for training)
without changes to the harness or this server.
