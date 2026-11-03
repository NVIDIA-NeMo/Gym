# NeMo Fabric Agent

Run interchangeable agent harnesses through [NeMo Fabric](https://github.com/NVIDIA/NeMo-Fabric) inside NeMo Gym.

## Agent sessions

This adapter pins **NeMo Fabric 0.5.0** and implements the **Evaluation** capability level.
An Environment Server seeds resources, passes their `SandboxAccess` to Fabric, invokes
the agent, closes it, and then verifies. Fabric runs only the agent in that sandbox;
resources retain task preparation, sandbox ownership, and scoring. AnySWE is not used.

Presets are unbound agent templates. Compose one with `single_agent_turn` (typed tasks)
or `single_agent_turn_legacy` (prepared flat rows), an existing Resources Server, and
a Model Server. Set `resources_tool_transports: []`. See the
[session composition example](../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx#fabric-agent-sessions).
No benchmark allowlist is expanded by this integration; use
`--allow-unsupported-pairing` for exploratory pairings that require it.

- Borrowed sandboxes are disconnected after confirmed harness termination and artifact
  collection; Resources destroys them. For tasks without a sandbox, configure
  `sandbox_provider` and `sandbox_config` and the agent creates and stops its own.
- Seed installs a private CPython 3.13.14 + the selected Fabric extra under
  `/opt/nemo-gym-fabric`. It uploads a checksum-verified standalone Python from the
  controller and installs dependencies in that interpreter, leaving task Python,
  `PATH`, `HOME`, and worktree dependencies alone. Linux x86_64/aarch64 with glibc,
  bash, tar, gzip, sha256sum, flock, writable `/opt` and `/tmp`, and outbound pip
  dependency access are required. Healthy cached runtimes are reused. Adapter-specific
  external CLIs still need to be available in the sandbox; all Fabric adapters are
  not automatically qualified by this bridge.
- `num_workers: 1` is required. A session supports one activation; identical retries
  join it. Closing terminates the shared supervisor before releasing the connection.
- Sessions accept text conversations and instructions. Nondefault per-request sampling,
  token limits, response/tool schemas, image inputs, replayed tool histories, and
  continuation are unsupported. Unsupported controls fail explicitly. Select model and
  harness budgets in the agent/Fabric config; the deadline also respects
  `metadata.agent_timeout_sec` and the seeded task limit.
- Required MCP/direct HTTP grants, task-specific agent users, static MCP/skills,
  additional model roles, and custom session artifact/environment paths are rejected.
  Ignored optional grants are reported as observation gaps.
- The rollout-qualified Gym model URL is preserved. Close returns cleanup evidence
  and explicit gaps for invocation ownership and compaction; token-ID training and
  Observability conformance are not claimed. A killed worker may leave no final
  Fabric transcript; timeout responses remain incomplete with a declared gap.
  In Fabric 0.5, mini-SWE without Relay returns only the final output and API-call
  count. Its missing intermediate transcript and token usage are declared as gaps;
  raw Model Server capture remains the evidence for individual model requests.

The existing host `/v1/responses` and resource-bound `/run` remain available for
compatibility. `/run` requires an explicit `resources_server` binding.

Fabric and directly wrapped harnesses can differ in SDK/CLI versions, system prompt
composition, permissions, tools and telemetry. Keep those choices in an A/B run's
receipt; sharing an adapter name does not establish behavior or performance parity.

`configs/` has a preset per Python adapter: Claude, Codex, Deep Agents, Hermes, mini-SWE-agent, OpenClaw, and
OpenHands. OpenClaw also needs its CLI on `PATH`: `npm install --global openclaw@2026.9.4` (Node.js 24.16 or later).

## Relay

Set `relay: true` to record each run with [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). The response is then
built from Relay's ATIF trajectory, unless the adapter's own output has more tool outputs or reasoning (Relay's ATIF
records no reasoning), and `fabric_result.relay_atif` holds the trajectories, since the Relay files are removed
after the host `/run` path. Session responses use the same ATIF/native-output selection, but do not expose
`fabric_result` as a verifier field. On the host path, adapters without Relay support, and Hermes without a Hermes Agent build that supports Relay, run
without it. Relay is experimental: in NeMo Fabric 0.5.0 several adapters record tool calls without their outputs,
and mini-SWE-agent runs fail more often with it. Claude and Codex record through a Relay gateway that accepts a
plain-HTTP model URL only on loopback, so give them an HTTPS or loopback model URL.
For session runs Relay must be supported by the selected adapter; failures are surfaced.
Leave it disabled for the first native/Fabric comparison.
