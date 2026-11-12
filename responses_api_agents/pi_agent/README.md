# Pi Agent

Pi runs the [Pi CLI](https://github.com/earendil-works/pi) inside the task sandbox supplied
by the benchmark's Resources Server. The Environment Server coordinates setup, agent execution,
verification, and cleanup. Pi borrows the sandbox and runs its own model/tool loop there.

[configs/pi_agent.yaml](configs/pi_agent.yaml) is the default agent configuration. Benchmark
and agent settings are independent; the run configuration connects them through the
Environment Server.

Training token-ID capture is off by default. With `model_server`, model calls go to the Gym Model Server under
the rollout's prefix, so capture needs only run-level `token_id_capture.enabled: true` and either the agent's
`token_id_capture: true` or run-level `token_id_capture.all_agents: true`.

## Configure and run

The benchmark owns task data, preparation, verification, and task sandbox settings.
The harness owns its runtime and model/tool loop. The Environment Server binds the two
and closes the agent before verification.

Run from the Gym repository root with Gym and the benchmark's preparation dependencies
installed. For SWE-bench Pro, save this composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/pi_agent/configs/pi_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: pi_agent
      resources_tool_transports: []
```

Supply the `policy_model` Gym Model Server, `policy_model_name`, and `sandbox` provider in
`model-provider.yaml`. The agent's `model` defaults to `${policy_model_name}` and remains
overridable. The sandbox must be able to reach the Model Server.

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent pi_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

Environment Server is the default entry point for rollout execution. With the single-agent
session-based configuration, the collector calls Environment Server `/run`: seed Resources,
seed the agent, call its rollout-prefixed `/v1/responses`, close the agent, verify, then close Resources. Prepared
flat rows use `single_agent_turn_legacy`, an input/output adapter over `single_agent_turn`.
This composition is benchmark-independent: Environment Server owns the episode sequence.
For tasks with a Resources-owned sandbox, the harness borrows that sandbox.
The adapter converts row formats; execution uses Resources/Agent session APIs. Collection
does not call the agent's compatibility `/run`. No additional materializer is needed.
Pass the same configuration to startup and `--no-serve` collection; collection does not
inherit routing settings from the running servers.

### Switch harness or benchmark

To change a compatible harness, replace its config import, harness-specific settings,
the Environment Server's `agent_server.name`, and the collection command's `--agent`.
Keep benchmark data, preparation, and verifier settings unchanged. To change a compatible
benchmark, replace its Resources config/reference and prepared input, keeping the harness
definition unchanged. Check tool grants, task-image/runtime support, model API, and the
benchmark's declared `allowed_agents` before running a new pairing.

Use this explicit composition for now. `--agent` selects a configured agent; it does not
install or rebind one. The existing `--agent-type` swap and automatic benchmark-data lookup
still depend on legacy Agent-to-Resources bindings; they are not equivalent to this workflow.
A new pairing does not need another combined preset.

`configs/profiles/benchmark.yaml` is a swappable profile for benchmarks whose Resources server supplies no sandbox,
such as HLE. It leaves `resources_server.name` unset (`???`), so Gym composes this agent onto the benchmark's own
agent instance, and each session creates its own sandbox through the `sandbox` provider. That sandbox must reach the
model server by address, so the profile sets `use_absolute_ip: true` and keeps the head server and the agent itself
on `127.0.0.1`: the head server's config endpoint is unauthenticated, and an unseeded request to the agent runs the
harness on the Gym host. The profile sets the sandbox's lifetime (`ttl_s`), which OpenSandbox otherwise ends after
10 minutes. Supply the image in `sandbox_config.image`; set other `sandbox_config` keys under the profile's own
instance name, `pi_agent.responses_api_agents.pi_agent.sandbox_config`. Like `pi_sandboxed_agent`'s profile, it sets
`network_access: model_only`, so the sandbox reaches only the model server; a search preset overrides it with
`model_and_tools`. A restricted sandbox cannot download Pi, so supply `sandbox_runtime_dir`, or set it to `null` for
an image prebaked at the default path.

HLE, HLE Diamond, GPQA and APEX Shortlist need no Environment Server setting: their judges serve no tools, so their
Environment Servers grant the agent none. This agent rejects a required `direct_http` grant with a 422, so a
benchmark whose Resources server serves its tools over direct HTTP does not pair with it.

## Runtime and model requirements

Use an exact `pi_version` (default **0.80.2**), and a Gym `model_server`.
For a Resources-owned task sandbox, supply direct `SandboxAccess` with an absolute
task working directory. Without access, configure agent-owned creation as noted below.
Session setup rejects `pi_version: latest`. Pi's `resources_server` setting is required
only for its compatibility `/run`, not agent sessions.

Supported task images are Linux x86_64/aarch64 glibc or x86_64 musl/Alpine with
bash, tar/gzip, and SHA-256 utilities; no Python is needed. Session setup uses POSIX sh to install missing Bash
with apt-get or apk when running as root, before launching the Bash runtime installer;
otherwise preinstall it in the image. Pi runs under Gym's shared Bash harness runner and process supervisor. Setup fails on either a nonzero exit or a provider error.
Older Alpine images
also need patchelf so Pi's Node can use a private, checksum-verified C++ library without
replacing the task's system library. The pinned Node version has no arm64 musl build.
The provider must support sandbox exec and file upload/download.
Installation needs network access to nodejs.org and npm; musl also uses unofficial-builds.nodejs.org
and, for the older C++ runtime fallback, dl-cdn.alpinelinux.org. An image without that access can carry the
runtime instead: set `sandbox_runtime_dir` to the directory `install_pi_runtime.sh` produces, copied whole
(`node/`, `pi/` and, on older Alpine, `libstdcpp/`), and setup checks that it runs `pi_version` rather than
installing one. A runtime that is missing or runs another version refuses the seed with HTTP 422.

The Node runtime, Pi package, and private `PI_CODING_AGENT_DIR` are outside the task repository.
Task HOME and XDG directories are preserved for tools and verification. Pi's
Node is addressed by absolute path; it does not replace the task's Python or Node on PATH.
Sandbox sessions bound each bash command and advise the agent to search repository/local
dependency directories rather than shared mounts. This returns control after a stalled tool;
the separate `timeout` still bounds the whole agent episode. Installer errors include both
stdout and stderr so provider status messages do not hide the actual failure.
Project extensions, skills, prompt templates, and themes are disabled; repository context
files may still be read by Pi.

Pi provides its own sandbox tools. It can also use the Resources server's tools through MCP tool grants:

- Set `expose_tools_over_mcp: true` on the Resources server and `resources_tool_transports: [mcp]` on the
  Environment Server. Pi speaks only the stateless JSON MCP endpoint that Gym's Resources servers expose,
  and the sandbox must reach the grant's URL, as it must reach the model server; both go through any HTTP
  proxy the sandbox environment sets.
- `gym_mcp.mjs` names each tool `<grant name>_<tool name>`, at most 64 letters, digits, `-` or `_`. Seeding
  refuses a required grant whose name is not 1 to 62 of these characters, and two required grants whose names
  differ by a `_<suffix>` (such as `web` and `web_v2`), since their tools could not be told apart. An optional
  grant with either problem is left out and recorded as an `mcp_server_skipped` observation gap. A tool name
  that does not fit is found only at setup in the sandbox.
- Tool discovery waits up to 30 seconds per request. Each tool call waits up to
  `sandbox_mcp_timeout_seconds` (default 300); a call that times out keeps running on the server, so set
  it above the slowest call the server makes. Tool errors from the server reach the model.
- A required grant that Pi cannot set up stops Pi before the first model call. The error and an
  `mcp_setup_failed` gap give the cause, naming the grant and the server's scheme, host and port when a grant
  failed. An optional one is skipped and recorded as an `mcp_server_skipped` gap.
- A successful call of a tool in `finish_tool_names` (its name on the server, such as `finish`) ends the run
  when every tool call of that turn is such a call; a turn that also ran another tool continues. The response
  and observations then count the run as completed. Pi may still make one compaction call after that turn.
  A finish call from a reply cut off at the output limit is not run, since its arguments may be cut off, and
  the model is asked to call it again. A reply counts as cut off when the server reports `length`, or when it
  used all of its `max_tokens`, as vLLM reports `tool_calls` for a reply it parsed a tool call from. Setup
  fails if no grant left in offers a listed tool. Names are matched without the grant, so another grant's tool of
  the same name would end the run too: as for grant names, an optional grant that offers a listed tool another grant
  offers is left out and recorded as an `mcp_server_skipped` gap, and two required grants that offer the same listed
  tool fail setup.
  Without usable MCP grants the setting has no effect.
- Responses name MCP calls `mcp__<grant name>__<tool name>`; a Resources server's `/verify` strips its own
  grant's prefix. Observations and model calls keep the names the model saw.
- The grant's headers are readable inside the sandbox, including by the model's tools. For a Resources
  server grant that is the episode's session token, which gives access only to that episode's tools.

Required Resources HTTP tools (`direct_http`) are rejected. Sandbox sessions also reject host command,
extra-argument, and environment overrides; those remain available on the local path.

## Requests and settings

Sandbox session input is one user message, optionally preceded by a system message. The user message
may include images as base64 data-URI `input_image` parts. The adapter stages each image at
`/tmp/nemo-gym-media/<sha256>.<extension>`, a path named by the image's content, and passes it to Pi as an
`@file` argument, so the same task gives Pi the same prompt in every sandbox. Staging is part of the
activation: it spends the agent timeout, and closing the session stops it. Staged images stay in a
borrowed sandbox after the session ends. A session declares image input on the Gym model when the prompt
has images or when the agent configuration sets `image_input: true`. A continuation session keeps declaring it
after any activation's prompt has images, because later activations resend those images from the saved
conversation. With image input declared, Pi also sends the
model images its tools return, such as task files it reads, and a text-only deployment rejects those requests.
Without it, Pi replaces such an image with a note saying it was omitted. Set `image_input: true` for a vision model
whose tasks keep their images in the sandbox, where the prompt has none.
Images in the system message and image URLs are rejected. Both sandbox
and local execution combine the configured system prompt, request `instructions`, and input
system message in that order. Sampling and chat-template settings belong on the Gym Model Server.
Sandbox sessions reject unsupported request controls before execution.

Pi decides how to send each attached file:

- Pi adds a `<file name="<sandbox path>">` tag to the end of the prompt text for each file, and sends the
  images after all the text, so the order of text and images in the request is not kept.
- Pi recognizes an image from its leading bytes, not from the declared media type. Pi inlines a file it
  does not recognize as an image, such as an animated PNG, into the prompt as text.
- Sessions turn off Pi's image resizing (`images.autoResize` in `settings.json`), so images reach the model
  as the environment sent them, both prompt images and images the read tool returns.
- The `detail` field of an `input_image` part has no effect.

Saved trajectories record the user prompt's text but not its images.

Set the agent configuration's `max_output_tokens` for a per-model-call output cap, including
reasoning tokens. Under the default `output_token_policy: fixed`, an adapter-owned Pi extension applies it as
`max_tokens`, preserving any smaller upstream cap; under `remaining_context` the extension removes the cap
(see [Local CLI compatibility](#local-cli-compatibility)). Limits must be positive JavaScript-safe integers. Request-level
`max_output_tokens` is rejected in sandbox sessions because a total-response budget is not implemented. Model Server
configuration must not replace the agent's cap with a larger value. Enforcement is tested with Pi 0.80.2.

## Lifecycle and ownership

1. Environment Server asks Resources to seed a task. Resources creates the sandbox.
2. Environment Server passes `SandboxAccess` to Pi's `/v1/agent_sessions` endpoint.
3. Pi connects as a borrower and automatically runs [install_pi_runtime.sh](install_pi_runtime.sh)
   to install Node 22.19.0 and the configured Pi package outside the task repository.
   This installs the harness runtime, not task dependencies or the test environment.
4. Environment Server calls the rollout-prefixed `/v1/responses` route with the agent-session cookie.
5. Pi runs in the sandbox, uses its built-in tools, and sends Chat Completions to
   the rollout-prefixed Gym model-server URL. The sandbox must be able to reach that URL.
6. Agent close confirms harness termination, returns observations,
   removes session files, and disconnects. Missing termination evidence keeps close retryable;
   a final negative receipt fails the activation but releases the borrowed connection.
7. Environment Server asks Resources to verify and close the task session. The benchmark
   owns its verification procedure and sandbox teardown.

The supervisor stops live harness-group processes. On timeout or close it also kills, within
the cleanup deadline, the harness's descendants in other process groups (typically tools in
their own sessions) and processes that inherited the harness's `NEMO_GYM_SUPERVISOR`
environment marker, such as services a tool daemonizes during the stop. Processes that left
the harness before the stop, typically services from completed tools, remain for the sandbox
owner to stop later.
Natural CLI exit cannot distinguish unfinished tools in other sessions from services;
those processes may still change task files. The stop-time cutoff also has PID races and
cannot classify later service forks. It is not a security boundary.
Services must use task-owned files rather than the temporary agent session directory.
The supervisor and the runner above Pi are bash scripts, so a task agent's
`pkill -f python3`, `pkill python3` or `killall python3` does not end Pi.

Harness execution uses `nemo_gym.agents.utils.sandbox_session.SandboxSession`, shared with Hermes.
The adapter stages its worker input and collects its own output. The shared session
uploads the supervisor at activation and owns its control files and log access; the common lifecycle
confirms process cleanup, captures artifacts, then releases the provider connection or
owned sandbox. Concurrent closes share that work; missing evidence or failed release remains retryable,
and interrupted activations keep captured observations after session files are removed.
`supervisor_client.py` handles controller-side launch/stop commands; `process_supervisor.sh`
runs inside the sandbox. Benchmark Resources session identity/verdict state stays separate.
Missing worker identity or exit diagnostics become `runtime_info_unavailable` or
`worker_exit_code_unavailable` observation gaps. Positive cleanup confirmation remains
required; the adapter validates the captured terminal events without inventing an exit code.

A sandbox session needs a matching episode and rollout identity. Without `continuation` in the seed, it
supports one activation, and Pi runs with `--no-session`, so it saves no conversation. Identical request retries join the running activation or replay its result;
changed requests are rejected. A disconnected HTTP caller does not cancel Pi; session close owns cancellation.

### Multi-turn continuation

A seed with `continuation` opens a session that takes later user turns. Pi saves the conversation under
the session directory with `--session-dir` and a session ID fixed at seed, and each activation resumes it,
so the model sees earlier turns, tool calls, tool results and images. Pi stores each image's data in the
saved conversation, so a later turn does not need an earlier turn's staged file. A seed that requires
`trajectory` observations is refused: Pi reports tool calls with timing, reasoning and compactions, but
builds no trajectory record.

- Each activation sends the `x-ng-agent-activation` header with its 1-based index. The base agent joins
  and replays a retry of the same index with the same input, and rejects a changed input, a skipped index,
  and a new index while one is running or after one failed.
- An activation is one user message, optionally preceded by a system message, with the same request
  controls as a one-activation session. Its images are staged as above, within the activation.
- Pi does not save its system prompt, and rebuilds it for the whole conversation on every run. Every
  activation therefore passes the first activation's system prompt. A later activation may repeat it or
  leave it out; a different one is rejected.
- Each activation stages its prompt, supervisor and output in its own directory, under its own timeout,
  which preparing the activation, waiting for a runner slot and staging also spend. An activation whose
  timeout runs out before Pi starts returns an `incomplete` response with no output, and its invocation
  reports error type `timeout`. Pi's saved conversation is as the previous activation left it, so the
  session can still be continued. Its model calls send `x-session-id`
  `<agent session ID>:activation-<index>`, the invocation ID its observations use, so the model server's
  capture correlates each call with its activation. Responses carry `activation_id` and
  `harness_session_id` metadata, and the activation's own observations in `_ng_agent_observations`.
- Each activation's invocation reports its status, its duration, and an error type: `timeout` or
  `cancelled` for a run stopped at its deadline or by close, `context_limit` for a run that reached the
  model's context limit, `provider_error` for a model call error Pi reported, and `server_error` for any
  other failure.
- Close stops a running activation and returns every activation's observations in order, with
  `cleanup_confirmed`.
- A turn stopped at its deadline can still be continued: when a stopped run leaves a tool call without
  a result, Pi 0.80.2 adds a "No result provided" result before the next turn.
With several agent workers, session routing sends each session's requests to the worker that seeded it.
Pi never falls back to a host CLI when sandbox setup
fails. Calls without an agent session retain the local Pi behavior; they do not operate
on the Resources-owned task sandbox.

Server startup does not install Pi on the host. Host installation happens only on the
first local invocation. Sandbox sessions install their runtime during session initialization,
using the same seed/close contracts as Hermes. The shell script is an internal implementation
detail, not a separate endpoint or a setup step users must run.

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: local compatibility launcher; task sessions use their installed Pi runtime
- `model`: the served model ID for the configured Gym Model Server
- `model_server`: Gym Model Server used to generate the Pi provider entry
- `context_window`: context limit for a generated model entry
- `max_output_tokens`: output limit for a generated model entry
- `env`: extra subprocess environment variables for local compatibility calls
- `workspace_root`: where per-request HOMEs are created and deleted
- `thinking`: passed to `--thinking` (off, minimal, low, medium, high, xhigh)
- `system_prompt`: appended via `--append-system-prompt`
- `timeout`: seconds for the `pi` run
- `extra_args`: extra Pi flags for local compatibility calls
- `models_config`: provider configuration for local compatibility calls
- `pi_version`: npm version to pin on install (null means latest on the local path; sandbox sessions require an exact version)
- `resources_server`: required only for the agent's existing `/run` endpoint, not for sandbox sessions or direct `/v1/responses`
- `sandbox_install_timeout_seconds`: sandbox runtime installation timeout (default 600)
- `sandbox_runtime_dir`: absolute sandbox path of a preinstalled runtime to use instead of installing one
  (default none)
- `network_access`: egress of a sandbox the agent creates: `inherit` (default, the provider's network),
  `model_only`, or `model_and_tools`; OpenSandbox only
- `sandbox_bash_timeout_seconds`: maximum runtime of each sandbox bash tool call (default 900);
  shorter tool-requested deadlines are preserved
- `sandbox_mcp_timeout_seconds`: maximum time of each call to a granted MCP tool (default 300)
- `finish_tool_names`: granted MCP tools whose successful call ends a sandbox session's run; only one required
  grant may offer each (default none; uses Pi's `terminate`, available from Pi 0.69.0)
- `session_close_timeout_seconds`: sandbox process cleanup timeout (default 60)

## Results and limits

Responses retain thinking text, tool calls/results, and Pi-reported usage, including cached
input tokens. Model limits and timeouts preserve gradable partial work after successful agent close.
Reported provider errors retain partial output and a classified cause for the Environment Server to decide grading and episode retries; independent runtime failures remain execution errors;
they must not be reported as incorrect solutions. When Pi's overflow recovery fails on its compaction
call, the run is reported as a provider error with an `unknown` cause, which makes it eligible for an
episode retry by default, even when that call itself hit the context limit. Close still returns
captured observations.
Pi does not set reward or masking policy.

Response metadata includes `harness_execution: sandbox`, hostname, supervisor PID, and
the pinned Pi version. When `timeout` stops a run that was still working and in which no call of a
`finish_tool_names` tool had succeeded, the metadata also has `stop_reason: wall_time`, as Stirrup reports a timeout,
so a benchmark can tell a timeout from a run that ended without a submission. A run has ended once a reply stops
(`stop` or `length`) without a tool call, even if `timeout` then stops Pi's compaction call, unless it is an empty
`length` reply at a full context window: Pi counts that as a context overflow and compacts to retry it (Pi 0.80.2 then
exits with code 1, which fails the run). A timeout while Pi waits to retry a failed model call, or while
the provider gives up on the supervisor after the deadline, is reported the same way. Model-call references and tool
observations are returned by agent close.
Tool timestamps are supervisor receipt times, not executor timestamps; unsupported evidence
is explicitly marked as gaps.

Session and close retries use Gym's shared agent-session implementation. Session state is
process-local. The Environment Server owns normal cleanup and the sandbox provider owns
expiry; Pi does not run a separate session-expiry timer. Unconfirmed termination blocks
verification. A final negative receipt releases the session for owner teardown; missing
evidence or failed release retains the handle for retry. Cleanup is cooperative, not a security boundary against sandbox code.

## Local CLI compatibility

`auto_compaction` and `output_token_policy` apply to local runs and sandbox sessions alike. Under
`output_token_policy`, `fixed` caps each request at `max_output_tokens`, and `remaining_context` drops the cap
so the model server sizes the output from the context the request leaves. The optional per-call `bash_timeout`
is local only; sandbox sessions use `sandbox_bash_timeout_seconds`. The dedicated
`pi_sandboxed_agent` also uses this local adapter inside its sandbox and supplies
`mcp_servers` for authenticated Gym tools. Sandbox sessions take Resources tools from MCP grants
instead and reject required HTTP tool grants rather than silently ignoring them.

Calls without an agent session retain local CLI compatibility. They do not use the
Resources-owned task sandbox. Configure a Gym `model_server`, or supply an explicit Pi provider
configuration as below. The self-contained math example keeps its local execution settings.

AnySWE, AnyTerminal, and HarnessAgent can continue forwarding their existing Responses
request fields. On this compatibility path, the configured model and per-call token cap
remain authoritative; forwarding request-level sampling fields or `max_output_tokens`
does not add support for them or enforce a total-response budget.

pi must be on PATH (auto-installed on the first local invocation, or `npm install -g @earendil-works/pi-coding-agent`).
Put `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
gym env start \
  --config environments/pi_math/config.yaml \
  --model-type openai_model

gym eval run --no-serve --agent pi_math_agent \
  --input environments/pi_math/data/example.jsonl \
  --output pi_rollout.jsonl --limit 5
```

Per request the agent writes `models.json` into an isolated `HOME`, runs one `pi` invocation with
stdin from `/dev/null`, then parses the jsonl `message_end` events. Example rollouts are in
`environments/pi_math/data/example_rollouts.jsonl`.

### Direct provider configuration

`model` is `<provider>/<model-id>`. Define the provider in `models_config` (written to
`~/.pi/agent/models.json`) and reference it here:

```yaml
model: nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct
models_config:
  providers:
    nvinf:
      baseUrl: ${policy_base_url}
      api: openai-completions
      apiKey: ${policy_api_key}
      models:
      - id: nvidia/qwen/qwen3-next-80b-a3b-instruct
        reasoning: false
```

For Gym-managed inference, set `model_server` to a Gym Model Server and set `model` to its served model id. The
agent creates the Pi provider entry automatically. Without `model_server`, the existing provider
configuration is unchanged.

Without `sandbox_access`, configure `sandbox_provider` and `sandbox_config`
(`SandboxSpec` fields such as `image`, `workdir`, and `ttl_s`) on the agent.
It creates a sandbox, runs there, and destroys it on close; the default workdir is
`/app`. A supplied access always wins, including its workdir; connection failure
never triggers a replacement or host execution. With neither access nor a usable
provider, setup fails. Verifiers that inspect task files must keep using a
Resources-owned sandbox, since agent-owned sandboxes are gone before verification.

`network_access: model_only` limits a created sandbox's egress to the model server's host;
`model_and_tools` also allows the host of each MCP grant. Each allowed host is open on every port.
Under `model_and_tools`, an MCP grant at an address the sandbox cannot reach, such as loopback, refuses the seed
with a 422, even an optional one that `inherit` would skip; a required grant under `model_only` is refused too. The policy is
set when the sandbox is created, so a session that borrows a task sandbox rejects any value other than `inherit`,
and a `sandbox_config.provider_options.network_policy` cannot be combined with it. A created sandbox must reach
the model server by address (`use_absolute_ip: true`). A borrowed sandbox relies on its owner's network policy,
and the owner never learns which endpoints the agent needs: a deny-by-default owner must allow the model server
itself, and a profile that sets `model_only` cannot be reused for a benchmark that supplies task sandboxes.

A restricted sandbox cannot download Pi, so pair it with `sandbox_runtime_dir`, or with an image that already holds
the runtime at the default path, `/tmp/nemo-gym-pi-node-22.19.0-<pi_version>`, with its `ready` marker.

Allowing the model server's host opens every Gym server bound to that address, not only the model server. With
`model_only`, set `absolute_ip_server_types: [responses_api_models]` so that only the model server leaves loopback;
with `model_and_tools`, also list `resources_servers` for the tool servers.

The policy is only as strong as the OpenSandbox server's egress mode. In `dns` mode only name resolution is filtered,
so a hard-coded IP bypasses the policy; `dns+nft` also enforces it at the IP layer. Use `dns+nft` for full isolation
(see [OpenSandbox](../../fern/versions/latest/pages/infrastructure/sandbox/opensandbox.mdx#network-policy)).
