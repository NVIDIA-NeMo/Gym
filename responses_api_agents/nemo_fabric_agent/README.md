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

Composing a preset with an existing environment now takes this session path, so the task needs a sandbox from
Resources or an agent-owned `sandbox_provider`; a text-only task with neither is refused at seed. The preset-level
`*_environment_server` `legacy_agent` relays that ran Fabric on the host are removed. For the host `/run` path, bind
`resources_server` yourself and write a `legacy_agent` block, as `environments/nemo_fabric_deepagents_math` does.
Every preset ships smoke-test budgets (`timeout: 600`, and `max_turns` of 8 for mini-SWE-agent and 30 for Claude and
Hermes), so size them for the benchmark.

- Borrowed sandboxes are disconnected after confirmed harness termination and artifact
  collection; Resources destroys them. For tasks without a sandbox, configure
  `sandbox_provider` and `sandbox_config` and the agent creates and stops its own.
- Seed installs a private CPython 3.13.14 + the selected Fabric extra under
  `/opt/nemo-gym-fabric`. The controller downloads a standalone Python from GitHub (or reads
  `/tmp/nemo-gym-miniswe-assets`), verifies its SHA-256, and uploads it; pip then installs the
  pinned packages in that interpreter, leaving task Python, `PATH`, `HOME`, and worktree
  dependencies alone. Only the Python archive and the staged requirements file are
  checksum-verified: pip resolves the packages and their dependencies at install time without a
  constraints file, so two installs on different days can differ. pip runs with `--isolated`, which
  ignores `PIP_*` variables and user configuration, so the sandbox needs direct PyPI access.
  Linux x86_64/aarch64 with glibc, a writable `/opt` and `/tmp`, and these tools are required: bash,
  uname, tee, mkdir, tar, gzip, sha256sum, flock, sleep, mktemp, ln, rm, and an `mv` that supports `-T`.
  Healthy cached runtimes are reused. Adapter-specific external CLIs still need to be available in
  the sandbox; all Fabric adapters are not automatically qualified by this bridge.
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
  Fabric transcript; timeout responses remain incomplete with a declared gap. Fabric stops
  Claude and Codex ten seconds before Gym's deadline and they report a `timeout` error; that is
  also an incomplete response, which is verified. Token-ID capture (`token_id_capture`) is refused
  at seed, because sessions are evaluation-only.
  In Fabric 0.5, mini-SWE without Relay returns only the final output and API-call
  count. Its missing intermediate transcript and token usage are declared as gaps;
  raw Model Server capture remains the evidence for individual model requests.

The existing host `/v1/responses` and resource-bound `/run` remain available for
compatibility. `/run` requires an explicit `resources_server` binding. Upgrading to Fabric 0.5.0 changes
their failure policy: a turn or step limit (mini-SWE-agent incomplete, the Deep Agents recursion limit, Claude
`error_max_turns`) now returns an incomplete response that `/run` verifies and scores, where Fabric 0.2.0 raised. The
upgrade also reaches AnySWE, AnyTerminal and the `harness_agent` presets that embed this agent.

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
plain-HTTP model URL only on loopback, so give them an HTTPS or loopback model URL on the host path. In a session
the harness runs in the sandbox, where loopback is the container, so give them an HTTPS model URL.
For session runs Relay must be supported by the selected adapter; failures are surfaced.
Leave it disabled for the first native/Fabric comparison.
