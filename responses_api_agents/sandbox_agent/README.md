# sandbox_agent

Runs any Gym agent harness inside a sandbox.

## Modes

- agent_only_runner: imports another agent's `responses()` inside the sandbox via a small
  runner. The harness code executes unchanged, only its model base URL is patched to a
  sandbox-reachable address. No Gym servers in the sandbox.
- gym_runner: starts full NeMo Gym inside the sandbox and forwards the task to its
  agent `/run` endpoint. For wrapping environments without a clean responses/verify split.

## Verification

- Tasks with an external verifier (e.g. math) verify on a normal resources server,
  the sandbox is closed after the rollout.
- SWE tasks set `patch_workdir`: the agent captures the in-box `git diff` into response
  metadata as `model_patch`, and `anyswe` grades it hermetically in a fresh sandbox with
  the official SWE-bench evaluation scripts.
- Tasks graded inside the task container (terminal) set `grade_in_box: true`: right after
  the solve, the agent stages the eval files (kept out of the box during the rollout so
  the harness cannot peek at tests), runs the eval command, reads back the reward file,
  and reports the reward in response metadata for the verifier (`anyterminal`).

## Delegate runtime

The server tars `nemo_gym/` and `responses_api_agents/` at startup (small, data and tests
excluded) and unpacks it to `/gym_mount` in each sandbox. `setup_commands` install the
delegate's dependencies, for example `pip install nemo-gym` for the import chain plus the
harness CLI itself.

Per task images come from the request metadata key named by `image_from_metadata_key`.

Because the harness inside the sandbox talks to a standard Gym model server, this agent
composes with future model-server capabilities (e.g. token-ID buffering for training)
without changes to the harness or this server.

See `environments/terminal` and `environments/swe` for example wirings.
