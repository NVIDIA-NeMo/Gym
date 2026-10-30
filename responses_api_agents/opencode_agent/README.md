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
needs Linux and Bash; it does not need Python. Missing Bash can be installed with
apt-get or apk when running as root; otherwise preinstall it.
Online installation needs GitHub access, curl, tar, gzip and a Debian or RHEL CA
bundle. Missing required packages are installed with apt-get or apk as root;
non-root images must preinstall them. System `rg` is a best-effort optimization:
setup tries apt-get or apk as root, but only warns if it is unavailable. OpenCode
can then download ripgrep on demand; offline images should preinstall it, or pin one
(see [Owner extensions](#owner-extensions)). Cached matching OpenCode runtimes need no
runtime download. A pinned binary is never downloaded. Task dependencies are preserved.

Local and sandbox execution share `runtime.OPENCODE_VERSION` and the model-call
correlation plugin. The plugin is uploaded into each sandbox when model-call
observability is enabled. Sessions disable automatic updates and models.dev fetches.

The runner uses ordinary sandbox `exec`, with an internal deadline.
The supervisor stops live harness-group processes. On timeout or close it also kills, within
the cleanup deadline, the harness's descendants in other process groups (typically tools in
their own sessions) and processes that inherited the harness's `NEMO_GYM_SUPERVISOR`
environment marker, such as services a tool daemonizes during the stop. Processes that left
the harness before the stop, typically services from completed tools, remain for the sandbox
owner to stop later.
Natural CLI exit cannot distinguish unfinished tools in other sessions from services;
those processes may still change task files. The stop-time cutoff also has PID races and
cannot classify later service forks. It is not a security boundary.
The supervisor and the runner above OpenCode are bash scripts, so a task agent's
`pkill -f python3`, `pkill python3` or `killall python3` does not end OpenCode.
The CLI must exit before
its SQLite database is copied to the agent server and exported there; services must use task-owned files rather than the
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
`instructions`. Request `temperature` and `top_p` are applied to every OpenCode agent
(see [Owner extensions](#owner-extensions)). Other controls, images, history replay
and required external HTTP/MCP tools are rejected. `context_window` and
`max_output_tokens` are independent config settings; leave room for the prompt in
the served model's context. Request-level overrides are not silently ignored.
Session `opencode_config` permits only `permission` and `tools`, preserving Gym capture.
OpenCode turns a `tools` map into base permission rules and merges `permission` on top,
so a later `permission` rule such as `"*": "allow"` re-enables tools that `tools`
disabled. Set `tools: null` (or `{}`) to make `permission` the only rule source; the
adapter then omits `tools` and passes `permission` exactly as given.

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
| Request controls | Existing local compatibility behavior | `temperature` and `top_p` applied with OpenCode 1.17.11; other controls rejected explicitly |

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

Ordinary single-turn sessions keep their existing protocol. With `return_session_trajectory:
true`, their close also returns the session's trajectory (one turn per saved assistant reply,
compaction summaries excluded), so rollout collection joins captured calls to those turns and
the summary call stays on the compaction record. Without it, collection builds one turn per
captured call, and the summary call of a compaction is counted as a step too.

### Offline sandbox runtimes and native model identity

For a sandbox without runtime-download egress, set `sandbox_executable_path` to
an existing sandbox-visible OpenCode binary and `sandbox_executable_sha256` to its
SHA-256. Seed verifies bytes and version before copying it into the adapter cache.
`sandbox_path_prefix` adds executable directories for installation and
CLI/tool subprocesses while retaining the image PATH. These paths must already
exist; the sandbox owner supplies mounts or image contents.

`sandbox_model_id` selects the native OpenCode model ID. It defaults to the existing
`dummy_model` compatibility alias. Benchmarks can set the native ID to retain
family-specific prompts, tools and defaults behind a Gym model route; the route
still controls the actual provider/model. Native ID changes can affect scores.

## Owner extensions

These sandbox-session settings are off by default; unset, sessions behave as described
above. Local calls do not use them. Each is general-purpose: the sandbox owner (usually
the Resources Server that lends the sandbox) supplies the files, and the adapter stages,
checks or reads them without any benchmark-specific logic. Seed refuses an unusable
setting or owner-supplied input with `AgentSeedRejectedError`, a final 422, so the
episode fails once with the reason; a failure a retry can fix stays a 5xx. In the
sandbox, their checks, staging, import and export need only a POSIX shell and coreutils
(`sha256sum`, and `tar` for an archive); the agent server reads OpenCode's session store
and checks archive members itself. OpenCode's `export` and `import` get
`sandbox_path_prefix` ahead of `PATH`, as `run` does.

### Request sampling

A request's `temperature` and `top_p` reach every model call OpenCode makes for the
activation. OpenCode 1.17.11 gives each agent its own sampling and none inherits from
`build` (`title` fixes temperature 0.5), so the adapter sets both on all built-in agents:
`build`, `plan`, `general`, `explore`, `compaction`, `title` and `summary`. An agent
name OpenCode does not define would add an agent, so the list is version-specific: with
another `opencode_version` (swe_together pins 1.15.13), a request `temperature` or `top_p`
returns 422, as it did before the adapter applied request sampling.
OpenCode sends `temperature` only for models that declare the capability, so a request
temperature also sets `temperature: true` on the adapter's model entry. Without these
fields the generated configuration is unchanged. Non-finite values return 422.

### Owner-staged plugins

`plugins` lists absolute sandbox paths of OpenCode plugin files staged by the sandbox
owner. They load after Gym's own plugins (the observability header plugin when model-call
capture is on, then `task_environment.mjs`), in the given order, in every activation.
Seed checks that each path is a regular file and returns 422 otherwise. Their correctness
is the owner's responsibility: OpenCode logs a plugin that fails to load and continues,
so a plugin that matters must fail closed through its own receipt, as
`task_environment.mjs` does with `task-environment.ready`. The adapter's environment is
fixed, so plugins should read their settings from files beside them (`import.meta.url`).

### Pinned offline runtime bundle

The OpenCode binary is pinned with `sandbox_executable_path` and
`sandbox_executable_sha256` (see
[Offline sandbox runtimes and native model identity](#offline-sandbox-runtimes-and-native-model-identity));
a binary that fails its checks is a retryable setup error. ripgrep and the plugin
dependencies, which OpenCode would otherwise download at runtime, are pinned alike: an
existing file inside the sandbox (`sandbox_<item>_path`, for example on a mounted cache or
staged by the sandbox owner), plus `sandbox_<item>_sha256`, which is required with the
path and is the only way to set one. The adapter copies the file, hashes the copy and
uses only those bytes. A digest or health-check mismatch fails seed with 422. A missing
file, or a health check that times out or is ended by a signal (such as the OOM
killer's), is a retryable setup error.

| Item | Fields | Where it goes | How to build it |
| --- | --- | --- | --- |
| ripgrep | `sandbox_ripgrep_path`, `sandbox_ripgrep_sha256` | `XDG_CACHE_HOME/opencode/bin/rg` in the session's private directories, where OpenCode 1.17.11 looks after `PATH`; it stays off task shells' `PATH`. Must answer `--version`. Replaces the installer's best-effort package install | ripgrep 15.1.0, which OpenCode 1.17.11 would download (`ripgrep-15.1.0-<arch>-unknown-linux-musl`) |
| Plugin dependencies | `sandbox_plugin_dependencies_path`, `sandbox_plugin_dependencies_sha256` | Unpacked into `XDG_CONFIG_HOME/opencode` before launch by the sandbox's `tar`, once the agent server has checked a download of the verified copy: members must be files, folders or links that stay inside it, no member path or link target may pass through a link, and none may carry setuid, setgid or sticky bits; npm's `node_modules/.bin` links pass | See below |

With any plugin configured (the adapter always configures `task_environment.mjs`),
OpenCode 1.17.11 installs `@opencode-ai/plugin` into its config directory at every start
unless `node_modules` exists and `package-lock.json` lists every dependency in
`package.json`. Each session has a fresh private config directory, so without the bundle
every start downloads about 63 MB from the npm registry; offline, start stalls about
70 s before the install fails. To build the archive, start the pinned OpenCode once
online with a fresh `XDG_CONFIG_HOME` and any plugin configured, then archive that
directory's dependency files:

```bash
tar -C "$XDG_CONFIG_HOME/opencode" -czf opencode-1.17.11-plugin-dependencies.tar.gz \
  package.json package-lock.json node_modules
sha256sum opencode-1.17.11-plugin-dependencies.tar.gz
```

An `rg` already on the task `PATH` takes precedence over the pinned copy, as in stock
OpenCode.

### Session exports

`export_dir` is an absolute sandbox directory chosen by the sandbox owner, outside the
adapter's `/tmp/nemo-gym-opencode*` storage and outside the task workdir: seed refuses one
that is the workdir, inside it or above it (422 when the paths show it, a setup error
when only resolving links in the sandbox does). After harness termination is confirmed
and before the session directory is removed, each activation's capture writes there:
- `sessions.json`: the stored session tree, `[{"id", "parent_id"}]` in creation order,
  written first so the owner can tell a failed export from an absent session;
- `<session id>.json`: `opencode export <session id>` for the root and every child
  session, each written atomically. A stale copy is removed before each export;
- `<root id>.events.jsonl` (`<root id>.activation-<n>.events.jsonl` in continuation
  sessions): the run's `--format json` event log, when one exists.

The session tree comes from the session store that the agent server copies out for the
capture. Each session is exported by a command of its own, with the session's private
directories, from the task workdir, without the run's configuration, so no plugin loads
again. An export counts only if OpenCode exits 0 and writes a JSON object. Writing shares
the capture step's budget (`session_close_timeout_seconds`) and ends a tenth of it early;
the provider stops an export still running then. A failure, including an exhausted
budget, never fails the activation or blocks release: it is reported as an
`export_dir_write_failed` observation gap whose detail names each failed session and
why. Files the model's tools can reach are not a trusted
channel; pick a fresh directory. The owner can then grade OpenCode's own transcript
after agent close. In a sandbox the agent owns, the files last only as long as the
provider keeps the sandbox's files after close: a provider that deletes the sandbox
deletes them too.

### Session import

`session_import_path` is an absolute sandbox path where the owner may stage an OpenCode
session export. If the file exists at seed, the adapter runs `opencode import` with the
session's private directories, without plugins, from the task workdir with `PWD` set to
it: OpenCode binds an imported session to the importing process's directory, and a
session imported elsewhere runs its tools there and can hang. The agent server then reads
the imported session from a copy of the session store. The first activation then
continues that session with `--session`, and the imported messages are passed as
`previous_message_ids`, so they stay out of output, usage and the trajectory, as in
multi-turn continuation. A missing file leaves the session unchanged. An export OpenCode
cannot import, or one that does not yield exactly one root session, fails seed with 422;
an import that times out or is ended by a signal is a retryable setup error.

### Working directory

OpenCode takes its directory from `PWD` before the process working directory. `run`,
`export` and `import` all start in the task workdir with `PWD` set to it (for `run`, the
harness runner's `cd` sets it), so all three bind the session to the same directory.
