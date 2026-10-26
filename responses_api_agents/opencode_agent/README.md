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
bundle. Missing required packages are installed with apt-get or apk as root;
non-root images must preinstall them. System `rg` is a best-effort optimization:
setup tries apt-get or apk as root, but only warns if it is unavailable. OpenCode
can then download ripgrep on demand; offline images should preinstall it.
Cached matching OpenCode runtimes need no runtime download. Task dependencies are preserved.

Local and sandbox execution share `runtime.OPENCODE_VERSION` and the model-call
correlation plugin. The plugin is uploaded into each sandbox when model-call
observability is enabled. Sessions disable automatic updates and models.dev fetches.

The runner uses ordinary sandbox `exec`, with an internal deadline.
The supervisor stops live harness-group processes and reaps its children. On timeout
or close it preserves children already adopted before the stop, typically services from
completed tools, and drains newly adopted children within the cleanup deadline, including
tools in separate sessions. The sandbox owner stops the preserved services later.
Natural CLI exit cannot distinguish unfinished tools in other sessions from services;
those processes may still change task files. The stop-time cutoff also has adoption/PID
races and cannot classify later service forks. It is not a security boundary.
The CLI must exit before
SQLite observations are captured; services must use task-owned files rather than the
agent session directory, which close removes. Capture requires confirmed termination.
Missing evidence remains retryable; a final negative receipt skips capture and verification
but permits connection release. Successful close receipts are retained for `session_close_retry_window_seconds` (default 300).
Session state lives in the worker that seeded the session.
With several agent workers, session routing sends each session's requests to the worker that seeded it.
Provider TTL and the episode owner handle abandoned sessions. Process cleanup is not a hostile-code security boundary.

Tool shells retain the task HOME and XDG directories through the shell environment hook,
so home-directory deliverables survive agent close. The CLI keeps private HOME and XDG
directories for configuration and SQLite state, including its separate `~/.opencode` lookup.
Use absolute paths for OpenCode file tools and shell `workdir`; CLI-side tilde resolution uses
its private HOME. The adapter requires a fresh successful-initialization receipt from
the task environment hook before accepting a gradable result. A missing or invalid
receipt fails the activation even if the CLI ignored a plugin-loading error.

The session path accepts one text task, optional system/developer text and
`instructions`. Unsupported controls, images, history replay and required external
HTTP/MCP tools are rejected. `context_window` and `max_output_tokens` are independent
config settings; leave room for the prompt in the served model's context. Configure
sampling on the Gym model server. Request-level overrides are not silently ignored.
Session `opencode_config` permits only `permission` and `tools`, preserving Gym capture.

Text, reasoning, tool results, partial output and usage come from the persisted
session tree. Unavailable artifacts or usage details are reported as observation
gaps. Reported API errors retain their classified cause for the Environment Server to decide grading and retries; worker and cleanup failures remain execution errors; explicit limits can retain
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

## Multi-turn continuation

Seed an owned or borrowed sandbox session with `continuation: {mode: resume_harness_session}`.
Call the existing `/v1/responses` endpoint with an ordinary Responses request and
`x-ng-agent-activation: 1`, then increment the header for each new user message.
Use the session cookie returned by seed. Checkpointed calls require the
`/ng-rollout/<id>` URL prefix. When checkpointing is disabled, the prefix is optional.
Whenever supplied, it must match the seeded episode. For example, after saving the seed cookie:

```bash
curl -b cookies.txt -H 'Content-Type: application/json' \
  -H 'x-ng-agent-activation: 1' \
  -d '{"input":"Implement a first version","metadata":{"agent_timeout_sec":"60"}}' \
  http://localhost:8000/v1/responses

curl -b cookies.txt -H 'Content-Type: application/json' \
  -H 'x-ng-agent-activation: 2' \
  -d '{"input":"Correction: handle empty input","metadata":{"agent_timeout_sec":"60"}}' \
  http://localhost:8000/v1/responses
```

Only the new user input is sent on each call. The adapter resumes the same OpenCode
session, retaining HOME, OpenCode history and compaction state. Runtime installation
happens once at seed using the existing installer. Identical retries join or replay
the original response; conflicting inputs, skipped indices and concurrent new
activations fail. Input validation runs before an index is consumed. A timeout
before the first process launch leaves the next activation able to start the conversation.
Retry state is process-local and retained for the session lifetime, without crash recovery.

The existing `metadata.agent_timeout_sec` budget is capped by the configured timeout.
For continuation it includes directory preparation, queueing, input staging and the
OpenCode model/tool loop. Cleanup has its own configured bound. The response uses
ordinary `status` and `incomplete_details`; timeout and OpenCode permission-denial
evidence are recorded on the activation's invocation. Session instructions cannot
be replaced by later inputs.

Each response carries the activation's `_ng_agent_observations` and `_ng_trajectory`.
Output and usage include only new work. Each invocation records its `activation_id`
and `harness_session_id`; response metadata also reports `harness_session_id`.
Every trajectory turn carries its activation. The OpenCode header plugin namespaces
`X-Session-Id` by the staged activation identity, including
child invocations, so captured model calls join to the activation that made them.
Close fences new work and returns cumulative observations, trajectory and cleanup
confirmation. A final negative termination receipt fails the activation and skips
verification; close releases the borrowed connection with `cleanup_confirmed: false`.
The task sandbox remains available to its owner for verification or teardown.

Ordinary single-turn sessions keep their existing protocol and evidence shape.

### Offline sandbox runtimes and native model identity

For a sandbox without runtime-download egress, set `sandbox_executable_path` to
an existing sandbox-visible OpenCode binary and `sandbox_executable_sha256` to its
SHA-256. Seed verifies bytes and version before copying it into the adapter cache.
`sandbox_python_executable` selects an existing Python interpreter (default
`python3`); `sandbox_path_prefix` adds executable directories for installation and
CLI/tool subprocesses while retaining the image PATH. These paths must already
exist; the sandbox owner supplies mounts or image contents.

`sandbox_model_id` selects the native OpenCode model ID. It defaults to the existing
`dummy_model` compatibility alias. Benchmarks can set the native ID to retain
family-specific prompts, tools and defaults behind a Gym model route; the route
still controls the actual provider/model. Native ID changes can affect scores.
