# NeMo Fabric Agent

Run interchangeable agent harnesses through [NeMo Fabric](https://github.com/NVIDIA/NeMo-Fabric) inside NeMo Gym.

## Agent sessions

This adapter pins **NeMo Fabric 0.6.1** and implements the **Evaluation** capability level.
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
- Seed installs a private CPython 3.13.14 (3.14.9 for Hermes) + the selected Fabric extra under
  `/opt/nemo-gym-fabric`. The controller downloads a standalone Python from GitHub (or reads
  `/tmp/nemo-gym-miniswe-assets`), verifies its SHA-256, and uploads it; pip then installs the
  pinned packages in that interpreter, leaving task Python, `PATH`, `HOME`, and worktree
  dependencies alone. Only the Python archive and the staged requirements file are
  checksum-verified: pip resolves the packages and their dependencies at install time without a
  constraints file, so two installs on different days can differ. pip runs with `--isolated`, which
  ignores `PIP_*` variables and user configuration, so the sandbox needs direct PyPI access.
  Linux x86_64/aarch64 with glibc, a writable `/opt` and `/tmp`, and these tools are required: bash,
  uname, tee, mkdir, tar, gzip, sha256sum, flock, sleep, mktemp, ln, rm, and an `mv` that supports `-T`.
  Hermes v0.22.7 uses its own dependency pins and overrides from `hermes_env/pyproject.toml`,
  installed with uv 0.13.4 in the private Python 3.14.9 runtime. This isolates its newer
  dependencies from the host server. Healthy cached runtimes are reused. Adapter-specific
  external CLIs still need to be available in the sandbox; all Fabric adapters are not automatically qualified by this bridge.
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
  Mini-SWE without Relay returns only the final output and API-call
  count. Its missing intermediate transcript and token usage are declared as gaps;
  raw Model Server capture remains the evidence for individual model requests.

The existing host `/v1/responses` and resource-bound `/run` remain available for
compatibility. `/run` requires an explicit `resources_server` binding. Upgrading from Fabric 0.2.0 changes
their failure policy: a turn or step limit (mini-SWE-agent incomplete, the Deep Agents recursion limit, Claude
`error_max_turns`) now returns an incomplete response that `/run` verifies and scores, where Fabric 0.2.0 raised. The
upgrade also reaches AnySWE, AnyTerminal and the `harness_agent` presets that embed this agent.

Claude failures with a structured upstream `api_error_status` retain that status in the session's
`ng_agent_outcome`; the Environment Server applies its grading/retry policy. Host execution also returns
permanent model-request outcomes, but keeps ungradable failures such as 429/503 on its exception path
because host wrappers verify directly. Unclassified setup/worker failures still raise.

Token usage follows Fabric's `nemo_fabric.detailed_token_usage.v1` inclusion flags. Claude's cache reads
and cache writes are added to its exclusive input count exactly once; inclusive provider counts stay
unchanged. Missing required dimensions leave usage unknown instead of reporting a subtotal as complete.

Fabric and directly wrapped harnesses can differ in SDK/CLI versions, system prompt
composition, permissions, tools and telemetry. Keep those choices in an A/B run's
receipt; sharing an adapter name does not establish behavior or performance parity.

`configs/` has a preset per Python adapter: Claude, Codex, Deep Agents, Hermes, mini-SWE-agent, OpenClaw, and
OpenHands. OpenClaw also needs its CLI on `PATH`: `npm install --global openclaw@2026.9.34` (Node.js 24.22.1 or later).

## Relay

Set `relay: true` to record each run with [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). The response is then
built from Relay's ATIF trajectory, unless the adapter's own output has more tool outputs or reasoning (Relay's ATIF
records no reasoning), and `fabric_result.relay_atif` holds the trajectories, since the Relay files are removed
after the host `/run` path. Session responses use the same ATIF/native-output selection, but do not expose
`fabric_result` as a verifier field. On the host path, adapters without Relay support, and Hermes without a Hermes Agent build that supports Relay, run
without it. Relay remains opt-in. Historical Fabric 0.5.0 runs had incomplete tool-result
trajectories; those results do not qualify 0.6.1. Relay 0.11 permits a remote plain-HTTP
model URL for Claude/Codex only with `harness_settings.insecure_http_model_hosts`.
Use HTTPS or explicitly allow the Model Server host; sandbox loopback names the container.
For session runs Relay must be supported by the selected adapter; failures are surfaced.
Leave it disabled for the first native/Fabric comparison.

For host Hermes runs, `adapter_project: hermes_env` installs a separate Python 3.14
worker environment on first use, with the published v0.22.7 Git revision. Concurrent
first calls share installation; failures propagate and a later call can retry. This
requires `uv` on the host. Session-only servers do not install a host Hermes runtime.
Neither the host Hermes environment nor sandbox runtime dependencies have a lock;
only their declared direct pins and overrides are fixed.

The Fabric 0.6.1 Hermes adapter itself redirects `HOME` as well as `HERMES_HOME`
into its artifact directory. Shell tools can inherit that home. This bridge does
not override upstream behavior, and Hermes SWE/home-directory compatibility is
not qualified by the text-task smoke.

For mini-SWE on a model without LiteLLM price metadata, use
`fabric_config.models.default.settings.cost_tracking: disabled` and a positive
`fabric_config.runtime.max_turns`. Fabric 0.6.1 no longer inherits
`MSWEA_COST_TRACKING`; the shipped preset uses the new setting and keeps its
8-turn budget. Migrate custom configurations that used the old environment flag.
