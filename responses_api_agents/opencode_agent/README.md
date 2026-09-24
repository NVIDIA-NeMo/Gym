# OpenCode Agent

This package supports native EnvironmentServer sessions (`execution_mode: sandbox`),
a local runner, and an explicit legacy compatibility mode. The existing
`opencode_sandboxed_agent` package and its benchmark configurations remain available.

For native sessions, compose [opencode_native_agent.yaml](configs/opencode_native_agent.yaml)
with [the TB4 episode profile](../../benchmarks/terminal_bench_4/episode.yaml).
Resources provisions the task; OpenCode borrows its sandbox, installs pinned
OpenCode 1.17.11, and executes there as the task user. The runtime needs Linux,
Python >=3.9, bash, and glibc. Automatic curl/CA installation requires root and apt-get;
otherwise the image must provide them or the deployment must stage the binary.

The adapter accepts one user text prompt and optional system/developer instructions.
It uses the rollout-prefixed Gym model endpoint. `max_output_tokens` is a per-call
cap, defaulting to the agent config; `max_steps` optionally bounds the build agent's
iterations. Unsupported sampling/history options fail before activation. Task user,
deadline, skills, and sandbox-local MCP connections come from `AgentTaskContext`.

One session permits one activation. A Linux subreaper confirms process cleanup;
close removes session files and disconnects before Resources verifies. Cleanup
failures retain the session for retry and block verification. Persisted trajectories,
model/tool observations, and available usage counters are returned through the
existing Gym response and close contracts.

The native implementation is adapted from Gym PR #3639. Local and legacy modes
retain their separate compatibility paths. Inference smokes do not establish
training support or full benchmark coverage.
