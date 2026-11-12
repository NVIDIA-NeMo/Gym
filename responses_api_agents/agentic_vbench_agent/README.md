# Agentic-VBench agent

Runs one [Agentic-VBench](https://github.com/PhiloLabs/agentic-vbench) task per rollout
through the benchmark's own harness (Harbor 0.6.6 + OpenCode 1.14.39) inside the task's
container and returns the reward the task's own verifier wrote. OpenCode's model calls go
through Gym's policy model server, so every call is captured per rollout; the agent never
re-scores a trajectory and never reimplements a task as question answering.

See the [benchmark README](../../benchmarks/agentic_vbench/README.md) for the measurement
contract, run-level knobs (`agentic_vbench_*`), backend requirements (Docker, rootless or
remote Podman) and judge credentials. Module map:

| Module | Role |
| --- | --- |
| `app.py` | Gym agent server: run request to Harbor trial, reward and `equal_family_reward` |
| `core.py` | Pinned benchmark inventory, task selection, artifact interpretation |
| `harbor_runner.py` | Harbor invocation, OpenCode config, timeouts, start-up probe retry |
| `remote_environment.py`, `podman_environment.py`, `remote-compose.yaml` | Harbor environment for rootless or remote Podman |
| `hub_judge.py` | Optional Repurpose rubric judge over the NVIDIA Inference Hub |

Tests: `gym env test +entrypoint=responses_api_agents/agentic_vbench_agent`.
