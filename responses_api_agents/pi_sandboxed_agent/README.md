# Pi Sandboxed Agent

> [!WARNING]
> **Deprecated.** `pi_sandboxed_agent` runs each episode through its own legacy `/run` path, behind a
> `legacy_agent` environment server, and will be removed after the environment-server migration.
> Do not build new benchmarks, agents or configs on this harness. Use
> [`pi_agent`](../pi_agent/README.md#configure-and-run) with sandbox sessions under an environment server instead.
> Existing configs keep working; the agent logs a deprecation warning when it starts.

Runs the native Pi CLI inside a provider-managed sandbox, with Gym handling model
requests, MCP tools, transcript export, and verification.

See the shared [OpenCode and Pi setup guide](../opencode_sandboxed_agent/README.md#offline-scientific-evaluation-with-opencode-or-pi)
for configuration, credentials, token and tool-call limits, network policies,
failure handling, and image builds.
Benchmark-specific instructions are in [HLE](../../benchmarks/hle/README.md#sandboxed-agents)
and [APEX Shortlist](../../benchmarks/apex_shortlist/README.md#sandboxed-agents).
