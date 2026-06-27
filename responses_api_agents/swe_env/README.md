# swe_env

Shared library for provisioning and grading SWE (software-engineering) task
environments. It is imported by `anyswe_agent` (and usable by any other Gym
agent that self-drives inside a SWE task sandbox); it is **not** a runnable
server and has no config or entrypoint.

Everything is provider-neutral, running over the `nemo_gym.sandbox` providers
(docker / apptainer / opensandbox):

- **`harnesses/`** — per-dataset-family recipes (SWE-bench, SWE-bench
  Multilingual, SWE-bench-ext, R2E-Gym, SWE-rebench, NV-internal). Each builds
  the task sandbox spec, materializes the model patch, runs the evaluation, and
  grades the result host-side via the official per-repo parser (falling back to
  a generic parser only where the official one is unavailable).
- **`sandbox.py`** — async sandbox lifecycle (`AsyncSweEnvironment`,
  `acquire_sandbox`) with always-teardown semantics.
- **`self_drive.py`** — provision a writable sandbox, inject a sandbox-reachable
  model endpoint / egress env, run an opaque agent launch command, and extract
  the resulting `git diff` patch.
- **`verify_task.py`** — grade a patch inline in a fresh sandbox (no separate
  `/verify` server), returning a mask-aware reward.
- **`parsing/`** — test-log parsers (relocated verbatim from `swe_agents`).

## Tests

The unit tests run against a scripted fake `SandboxProvider`, so they need no
Docker/apptainer and execute in CI:

```bash
ng_test +entrypoint=responses_api_agents/swe_env
```
