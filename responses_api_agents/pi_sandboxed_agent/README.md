# Pi Sandboxed Agent

> [!WARNING]
> **Deprecated.** `pi_sandboxed_agent` runs each episode through its own legacy `/run` path, behind a
> `legacy_agent` environment server, and will be removed after the environment-server migration.
> Do not build new benchmarks, agents or configs on this harness. Use
> [`pi_agent`](../pi_agent/README.md#configure-and-run) with sandbox sessions under an environment server instead.
> The HLE search presets (`pi_search.yaml`), which use `tool_servers`, have no session path yet and stay on this
> agent until they do. Existing configs keep working; the agent logs a deprecation warning when it starts.

Runs the native Pi CLI inside a provider-managed sandbox, with Gym handling model
requests, MCP tools, transcript export, and verification.

For a pinned baseline, set an exact `pi_version`. Each run checks the preinstalled CLI;
if the default `pi` command is absent, it uses Pi's existing sandbox installer to install
that version with private Node 22.19.0. Installation needs sandbox network access and the
[same image prerequisites as Pi sessions](../pi_agent/README.md#configure-and-run).
Restricted sandboxes require a matching preinstalled CLI or `sandbox_runtime_dir`.
A failed version check or installation fails the request and still requests sandbox cleanup.
When Resources supplies the sandbox, `/run` calls its `/close_session` after verification
or setup failure, then disconnects Pi's client. Pi stops sandboxes it allocates itself.
Cleanup has a bounded timeout, survives request cancellation, and fails the request if
teardown cannot be confirmed. Resources retains its failed close state for retry.

With `output_token_policy: fixed`, the baseline uses the session path's per-call output
cap and tool runtime guards. `remaining_context` remains available for existing presets.
Pi configuration stays in a private directory through `PI_CODING_AGENT_DIR`, while tools
keep the task's `HOME`. Resource-provided working directories are preserved.

See the shared [OpenCode and Pi setup guide](../opencode_sandboxed_agent/README.md#offline-scientific-evaluation-with-opencode-or-pi)
for configuration, credentials, token and tool-call limits, network policies,
failure handling, and image builds.
Benchmark-specific instructions are in [HLE](../../benchmarks/hle/README.md#sandboxed-agents)
and [APEX Shortlist](../../benchmarks/apex_shortlist/README.md#sandboxed-agents).
