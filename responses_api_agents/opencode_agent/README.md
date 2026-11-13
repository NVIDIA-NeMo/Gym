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
and apply to local calls too. The profile also denies the `webfetch`, `websearch` and
`skill` tools. Subagents (`task`) are intentionally enabled for this
general-purpose integration; the separate `opencode_sandboxed_agent` benchmark
profile still disables them. Use explicit `permission` overrides when comparing scores.
The command filters were written for SWE-style repositories: they also deny Git history
commands (`git reflog`, `git show`, `git log --all`, ...) and every `git fetch`, which some
terminal tasks need (for example recovering a lost commit or fetching a local bundle).
Allow those per run with `permission.bash` rules when a benchmark requires them.
Restrict tools with `permission` rules, not a `tools` map: OpenCode 1.17.11 places the keys
a `tools` map names before the permission map's `"*"` rule and applies the last matching rule,
so `"*": allow` then overrides every rule for those keys (naming `bash` voids every command filter).

Other flat-row benchmarks keep the `opencode_sandboxed_agent` entrypoint and configuration for their
Resources-owned lifecycle. HLE and APEX Shortlist run this agent from their plain-named OpenCode presets, with
`configs/profiles/offline_science.yaml`; their `*_legacy_agent` presets keep the deprecated agent.

The old `opencode_sandboxed_agent` package remains separate; only its shell PATH
quoting is fixed for directories containing spaces.

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

`network_access: model_only` limits an agent-owned sandbox's egress to the model
server's host; `model_and_tools` also allows the host of each MCP grant. Each
allowed host is open on every port. The OpenSandbox policy is set when the sandbox
is created, so a session that borrows a task sandbox rejects any value other than
`inherit` (the default, the provider's network), and it cannot be combined with
`sandbox_config.provider_options.network_policy`. Such a sandbox cannot download
the runtime: set `sandbox_executable_path` and `sandbox_executable_sha256`, or use an
image that already holds the runtime at `/tmp/nemo-gym-opencode-runtime-<opencode_version>`
(see [Offline sandbox runtimes](#offline-sandbox-runtimes-and-native-model-identity)).
Without either, the seed is accepted and each attempt then fails downloading OpenCode.
Preinstall `rg` too, since OpenCode otherwise tries to download ripgrep on demand.
Under `model_only`, a required MCP grant is refused at seed with a 422, since the
policy would block its host; under `model_and_tools`, a grant at an address the
sandbox cannot reach, such as loopback, refuses the seed. A borrowed sandbox relies on
its owner's network policy. The policy is only as strong as the OpenSandbox server's
egress mode: in `dns` mode only name resolution is filtered, so a hard-coded IP
bypasses it; `dns+nft` also enforces it at the IP layer and is needed for full isolation.

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
but permits connection release. A successful close fences the session and keeps its saved reply for `session_close_retry_window_seconds` (default 300).
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

The session path accepts one user task, optional system/developer text and
`instructions`. The user task may include images as base64 data-URI `input_image` parts,
alongside nonempty text. The adapter writes each image to `/tmp/nemo-gym-media/<sha256>.<ext>`
with `nemo_gym.sandbox.stage_media` and attaches it with `opencode run --file`. OpenCode shows
that path to the model, so naming the file by its content keeps the prompt the same across
rollouts of a task. In a borrowed sandbox these files remain after the session closes. OpenCode places the images, in order, before the task text and does
not use the `detail` setting. A session declares image input on the Gym model when the prompt has images or
when the agent configuration sets `image_input: true`. With image input declared, OpenCode also sends the model
images its tools return, such as task files it reads, and a text-only deployment rejects those requests. Without it,
OpenCode replaces such an image with an error text telling the model it cannot read images. Once an activation
declares image input, later activations of the session keep declaring it, because OpenCode resends earlier images
with the session history. Set `image_input: true` for a vision model whose tasks keep their images in the sandbox,
where the prompt has none. A session that declares image input also lifts OpenCode's image limits, so the model
gets each image's own bytes: by default OpenCode shrinks or re-encodes an image over 2000 px on a side or 5 MiB of
base64. A run on an endpoint with image limits can restore them with `attachment` in `opencode_config`, which
replaces the lifted limits; a limit it leaves out returns to OpenCode's default.
In a single-activation session, the timeout budget starts once the activation holds a
concurrency slot. Staging the activation's files, images included, spends it, OpenCode runs
for the rest, and a budget that runs out before launch fails the activation.
Images outside the user task and image URLs are rejected. Saved trajectories keep only the
text of each message, so they do not contain the images. Request `temperature` and `top_p` are applied to every OpenCode agent
(see [Owner extensions](#owner-extensions)). Gym agents read a request `max_output_tokens` as the
limit for each model call, as `simple_agent` does, and NeMo RL sets it on every row. It passes when
it is at least the configured `max_output_tokens`, OpenCode's limit for each call; a smaller value
returns 422. As in `simple_agent`, nothing limits the sum of output tokens over a session's calls. Other controls, history replay and
required direct-HTTP tool grants are rejected. MCP tool grants become OpenCode `remote` MCP servers, and the model sees
their tools as `<grant>_<tool>`. `mcp_tool_timeout_seconds` sets their connection and per-call
timeout; without it, OpenCode allows 30 seconds to connect and 60 seconds per call. A call still
fails after about 300 seconds, Bun's default `fetch` timeout. Before the first model call,
`required_mcp.mjs` checks
their status: a required grant that did not connect fails the activation, and an optional one is
reported as an `mcp_server_skipped` gap. A run that called the model but left no readable status
also fails, because nothing confirms its required tools; a run that failed for another reason keeps
that cause. `context_window` and `max_output_tokens` are independent
config settings; leave room for the prompt in the served model's context. Request-level overrides
are not silently ignored.
Session `opencode_config` permits only `permission`, `tools`, `attachment`, `compaction.auto`,
`compaction.reserved` and `agent.build.steps`, preserving Gym capture.
OpenCode turns a `tools` map into base permission rules and merges `permission` on top,
so a later `permission` rule such as `"*": "allow"` re-enables tools that `tools`
disabled. Set `tools: null` (or `{}`) to make `permission` the only rule source; the
adapter then omits `tools` and passes `permission` exactly as given.
Setting `compaction.auto: false` keeps OpenCode from
summarizing the session and continuing from the summary, in model calls that training cannot join
to the scored call; a run that fills the context then ends with a context-limit error.
`compaction.reserved` (a positive integer) is the token buffer OpenCode keeps below `context_window`:
automatic compaction starts after a call whose tokens reach `context_window` minus this buffer. Unset,
the buffer is the smaller of 20000 and `max_output_tokens`. It moves only that trigger;
`compaction.prune`, which rewrites old tool output, stays refused. A session refuses a buffer of
`context_window` or more, which would compact after every call.
`agent.build.steps` sets the step limit of the `build` agent, which the session runs, for each
prompt. From the call at the limit on, OpenCode adds an assistant message to each request that
asks for a text answer. It still offers the tools, so a model that keeps calling them runs past
the limit. OpenCode counts each pass of its session loop toward the limit, compaction passes included,
and compaction does not restart the count. A response whose prompt reached the limit carries
`metadata.harness_step_limit_reached: "true"`. Token capture recognizes this message
(`nemo_gym.models.token_capture.harness_notes`) and joins the calls after it to the earlier calls of
the rollout.

Text, reasoning, tool results, partial output and usage come from the persisted
session tree. Unavailable artifacts or usage details are reported as observation
gaps. Reported API errors retain their classified cause for the Environment Server to decide grading and retries; worker and cleanup failures remain execution errors; explicit limits can retain
gradable partial output. This path is eval-only; working inference does not establish
training token-ID/logprob support or benchmark accuracy. Sandbox observations use the shared
`FailureStage` vocabulary: execution is `agent`, while OpenCode’s own setup failure events use `seed`.
An existing failure stage is preserved when forwarded; scoring and retries belong to the caller’s outcome policy.

For local calls, OpenCode is installed on first use in a shared background-thread
task, without blocking the event loop. Failed installs can be retried. A configured `repo_dir` is
preserved; otherwise the temporary workspace is removed after the call. A local
working directory is not filesystem isolation. There is no bridge or execution-mode flag: a sandboxed benchmark
moves to this agent by changing its preset, as HLE and APEX Shortlist do.

`configs/profiles/benchmark.yaml` is a swappable profile for benchmarks whose Resources server supplies no sandbox,
such as HLE. It leaves `resources_server.name` unset (`???`), so Gym composes this agent onto the benchmark's own
agent instance, and each session creates its own sandbox through the `sandbox` provider. That sandbox must reach the
model server by address, so the profile sets `use_absolute_ip: true` and keeps the head server and the agent itself
on `127.0.0.1`: the head server's config endpoint is unauthenticated, and an unseeded request to the agent runs the
harness on the Gym host. The profile sets the sandbox's lifetime (`ttl_s`), which OpenSandbox otherwise ends after
10 minutes. Supply the image in `sandbox_config.image`; set other `sandbox_config` keys under the profile's own
instance name, `opencode_agent.responses_api_agents.opencode_agent.sandbox_config`. It sets `network_access: model_only`,
so the sandbox reaches only the model server; a search preset overrides it with `model_and_tools`. A restricted
sandbox cannot download OpenCode, so supply `sandbox_executable_path` and `sandbox_executable_sha256` for a binary
in the image.

`configs/profiles/offline_science.yaml` builds on `benchmark.yaml` for the offline scientific presets
(`benchmarks/hle/opencode.yaml`, `opencode_search.yaml`, `benchmarks/apex_shortlist/opencode.yaml`). It runs the
offline science image's pinned OpenCode binary with a 32,768-token output cap in a 229,376-token context, the
scientific-tools system prompt and subagents denied, and binds only the model server to the address sandboxes reach
(`absolute_ip_server_types`). The HLE README lists how it differs from the deprecated preset.

HLE, HLE Diamond, GPQA and APEX Shortlist need no Environment Server setting: their judges serve no tools, so their
Environment Servers grant the agent none. This agent rejects a required `direct_http` grant with a 422, so a
benchmark whose Resources server serves its tools over direct HTTP does not pair with it.

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
| Request controls | Existing local compatibility behavior | `temperature` and `top_p` applied with OpenCode 1.17.11; `max_output_tokens` (a per-call limit) accepted when at least the configured limit; other controls rejected explicitly |

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
activations fail. Input validation runs before an index is consumed. When the activation's
budget runs out before the first process launch, the next activation can still start the
conversation. A sandbox error while staging prompt images fails the activation instead.
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

By default (`return_session_trajectory: true`), the close of an ordinary single-turn session also
returns the session's trajectory (one turn per saved assistant reply, compaction summaries
excluded), so rollout collection joins captured calls to those turns, the harness evidence checks
match each model call to its step, and the summary call stays on the compaction record. With
`false`, collection builds one turn per captured call with no message ID, so every call is reported
as an unmatched assistant message and the summary call of a compaction is counted as a step too.

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
setting or owner-supplied input with `AgentSeedRejectedError`, a terminal 422, so the
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

### Repository project config

Sessions ignore the task repository's OpenCode project config and instruction files
(`AGENTS.md` or `CLAUDE.md`, `opencode.json`, `.opencode/`) unless `session_project_config`
is set. With it, the run reads them as OpenCode does when started in that repository; the
configuration Gym passes is merged last and still takes precedence. A repository can then
add instructions, agents, plugins or tools, so set it only for task repositories you would
run OpenCode in yourself. Export and import never read project config.

### Compaction summaries in the output

By default, a session's response output includes OpenCode's compaction summaries. A summary is
an assistant message like any other, so when a run ends on a compaction, a verifier that grades
the last assistant text gets the summary instead of an answer. With
`exclude_compaction_summaries: true`, the output leaves out each assistant message that
OpenCode marks `summary: true` or that its `compaction` agent wrote. The compaction record in
the observations keeps the summary text either way.

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
