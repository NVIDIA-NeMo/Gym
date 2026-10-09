# OpenCode Agent

`configs/opencode_agent.yaml` uses a Gym model server and supports sandbox
EnvironmentServer sessions and local CLI calls. An agent session selects the
sandbox; without one, requests run locally. Invalid or closed sessions fail.
Local `/run` requires a Resources binding; direct `/v1/responses` does not.

Training token-ID capture is off by default. Enable the agent's `token_id_capture`
only when explicitly collecting training data; evaluation observability remains independent.

The main config loads `configs/permissions/default.yaml` through `config_paths`.
It preserves the existing command restrictions; run configs can override individual
permissions under `opencode_agent.responses_api_agents.opencode_agent.opencode_config.permission`.
These command filters are not network or filesystem isolation.
They are adapted from Gym's benchmark runner restrictions, not OpenCode defaults,
and apply to local calls too. Subagents (`task`) are intentionally enabled for this
general-purpose integration; the separate `opencode_sandboxed_agent` benchmark
profile still disables them. Use explicit tool overrides when comparing scores.

Existing flat-row benchmarks keep the `opencode_sandboxed_agent` entrypoint
and configuration for their Resources-owned lifecycle.

The old `opencode_sandboxed_agent` package remains separate; only its shell PATH
quoting is fixed for directories containing spaces. No benchmark recipes or
Resources allowlists are migrated by this integration.

## EnvironmentServer sessions

Compose this agent with a Resources server that supplies `SandboxAccess` and the
generic EnvironmentServer. For example, an already prepared SWE-Pro JSONL can use:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/opencode_agent/configs/opencode_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

environment_routing_mode: legacy
environment_server_name: single_agent_turn_legacy

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server: {type: resources_servers, name: swebench_pro_resources_server}
      agent_server: {type: responses_api_agents, name: opencode_agent}
```

Save as `run.yaml`, supply the model and sandbox provider settings in `env.yaml`,
and use a model endpoint reachable from inside the sandbox:

```bash
gym env start --config run.yaml --model-type openai_model

NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING=1 \
  gym eval run --no-serve --config run.yaml --model-type openai_model \
  --agent opencode_agent --input /path/to/prepared-swe-pro.jsonl \
  --output outputs/opencode-rollout.jsonl --limit 1
```

The explicit opt-in is needed because SWE-Pro retains its existing `allowed_agents`
list. It does not change benchmark defaults or establish accuracy. Prepared rows
must contain the full prompt, including requirements and interface; `task_source`,
when present, must identify the selected Resources server. This bridge accepts
existing flat rows but orchestrates agent sessions; it does not call agent `/run`.
No additional benchmark preset or taskset materialization is required.

EnvironmentServer seeds Resources, opens the agent session, calls the rollout-scoped
`/v1/responses`, closes the agent, verifies, and closes Resources. The supplied
sandbox is borrowed: the agent disconnects, while Resources owns destruction.
Without access, configure `sandbox_provider` and `sandbox_config` on the agent to
create an agent-owned sandbox; it is destroyed on close. Verifiers that inspect
task files must use a Resources-owned sandbox. Missing access and provider is an
error, never a fallback to local execution.

Session setup installs the pinned OpenCode runtime inside the sandbox. The image
needs Linux, Python 3.8+ with SQLite and `fcntl`, and Bash. Missing Python/Bash can
be installed with apt-get or apk when running as root; otherwise preinstall them.
Online installation needs GitHub access, curl, tar, gzip and a Debian or RHEL CA
bundle. The installer also ensures `rg` is on PATH so `glob`/`grep` do not download
ripgrep into every session. Missing packages are installed with apt-get or apk as
root; non-root images must preinstall them. Cached matching runtimes need no
download. Task dependencies are preserved.

Local and sandbox execution share `runtime.OPENCODE_VERSION` and the model-call
correlation plugin. The plugin is uploaded into each sandbox when model-call
observability is enabled. Sessions disable automatic updates and models.dev fetches.

The runner uses ordinary sandbox `exec`, with an internal deadline and a Linux
subreaper to clean detached tool processes. Close succeeds only after confirmed
cleanup; uncertain cleanup remains retryable and blocks verification. Successful
close receipts are retained for `session_close_retry_window_seconds` (default 300).
Session state is process-local; use one worker. Provider TTL and the episode owner
handle abandoned sessions. Process cleanup is not a hostile-code security boundary.

The session path accepts one text task, optional system/developer text and
`instructions`. Unsupported controls, images, history replay and required external
HTTP/MCP tools are rejected. `context_window` and `max_output_tokens` are independent
config settings; leave room for the prompt in the served model's context. Configure
sampling on the Gym model server. Request-level overrides are not silently ignored.
Session `opencode_config` permits only `permission` and `tools`, preserving Gym capture.

Text, reasoning, tool results, partial output and usage come from the persisted
session tree. Unavailable artifacts or usage details are reported as observation
gaps. Runtime/model errors fail before verification; explicit limits can retain
gradable partial output. This path is eval-only; working inference does not establish
training token-ID/logprob support or benchmark accuracy.

For local calls, OpenCode is installed on first use in a shared background-thread
task, without blocking the event loop. Failed installs can be retried. A configured `repo_dir` is
preserved; otherwise the temporary workspace is removed after the call. A local
working directory is not filesystem isolation. Existing sandboxed benchmarks keep
using `opencode_sandboxed_agent` directly; there is no bridge or execution-mode flag.

## Configuration compatibility

The shipped YAML now uses `model: ${policy_model_name}`, `thinking: true` (the class
default), a 10,800-second run timeout, and no Resources binding. Local `/run` callers
must supply `resources_server`; direct `/v1/responses` callers do not need one.
Class defaults and explicit local overrides remain available. Sandbox creation
requires an explicit provider when Resources supplies no sandbox. Installation and
close timeouts use `sandbox_install_timeout_seconds` (600) and
`session_close_timeout_seconds` (60).

Local request handling is preserved for existing callers; session requests use the
stricter, single-activation contract described above. The intentional differences are:

| Setting | Local CLI | Sandbox session |
| --- | --- | --- |
| System text | Prepended to the user prompt; request `instructions` is not applied | Config/system/developer text and request `instructions` form an instructions file |
| Provider name | `nemo` | `nemo_gym` (internal alias for the same Gym model server) |
| `env`, `extra_args`, custom `command` | Applied | Rejected at seed, not silently ignored |
| Request controls | Existing local compatibility behavior | Unsupported controls rejected explicitly |

Both paths honor configured `thinking` and declare interleaved `reasoning_content`.
Different prompt placement can affect results; this integration does not establish
local/session score parity or migrate existing benchmark defaults.
