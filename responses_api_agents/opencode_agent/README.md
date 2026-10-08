# OpenCode Agent

`configs/opencode_agent.yaml` uses a Gym model server and supports native
EnvironmentServer sessions and local CLI calls. An agent session selects the
sandbox; without one, requests run locally. Invalid or closed sessions fail.
Local `/run` requires a Resources binding; direct `/v1/responses` does not.

Existing flat-row benchmarks keep the `opencode_sandboxed_agent` entrypoint
and configuration for their Resources-owned lifecycle.

See [OpenCode Wiring](../../fern/versions/latest/pages/evaluation/opencode.mdx)
for configuration, runtime requirements, model limits, ownership and cleanup.
