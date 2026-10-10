# OpenClaw Agent

Runs OpenClaw CLI (`openclaw agent --local --json`).
OpenClaw runs its own tools internally.
Resources server is used for verifier.

Minimal, meant to be extended, and currently eval-only. 

## Quick start

OpenClaw is auto-installed on the first local CLI invocation. Starting an agent server for sandbox
sessions does not install or execute OpenClaw on the agent-server host.
Make sure `env.yaml` is also set.

```bash
gym env start \
  --config environments/openclaw_math/config.yaml \
  --model-type openai_model

gym eval run --no-serve --agent openclaw_math_agent \
  --input environments/openclaw_math/data/example.jsonl \
  --output openclaw_rollout.jsonl --limit 3
```

## Model configuration

The default config binds `model_server` to Gym's `policy_model` and uses
`${policy_model_name}`. The adapter generates the OpenClaw provider configuration
for both local CLI calls and sandbox sessions. Configure the endpoint and API key
on the Gym model server.

Existing local callers may still supply their own OpenClaw provider configuration
when `model_server` is unset.

Compared with the previous shipped YAML, the fixed `nvinf` provider/model block
is removed. `workspace_root`, `openclaw_agent_id`, `thinking`, `system_prompt`,
`extra_args`, `env`, and `concurrency` now use class defaults unless overridden.
`resources_server` is now `null`: local `/run` needs an explicit Resources binding.
Session install/close timeouts default to 600/60 seconds.

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: the OpenClaw command, split on spaces so a multi-word launcher works (e.g. `npx openclaw`)
- `model`: `<provider>/<model-name>` (see Model id)
- `model_server`: optional Gym model server used to generate the provider entry
- `context_window`: context limit for a generated model entry
- `max_output_tokens`: output limit for a generated model entry
- `workspace_root`: where per-request workspaces are created and deleted
- `openclaw_agent_id`: passed to `--agent`
- `thinking`: passed to `--thinking` (off, low, medium, high, ...)
- `system_prompt`: prepended to the user message
- `setup_timeout`: seconds for `openclaw setup`
- `timeout`: seconds for the `openclaw agent` run
- `model_timeout_seconds`: request and stream-idle timeout for the generated Gym model provider
  (default: 600 seconds). OpenClaw still caps this by the overall agent deadline. Explicit provider
  `timeoutSeconds` settings in local-mode `openclaw_config` take precedence.
- `extra_args`: extra flags appended to `openclaw agent`
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `openclaw_config`: deep-merged into the generated `openclaw.json`
- `openclaw_version`: exact version to pin on install (npm ranges such as `^2026.9.0`
  are rejected); overridden by the `OPENCLAW_VERSION` env var, and falls back to
  `setup_openclaw.DEFAULT_OPENCLAW_VERSION`. An already-installed `openclaw` is only
  reused when `openclaw --version` reports exactly the resolved version; anything else
  is reinstalled on first local use after a restart. After an install, the launcher
  selected on `PATH` must report the requested version, or that local request fails.
  Note: releases ≥ 2026.9.0 store session history in SQLite instead of JSONL, which
  the agent's transcript reader does not support yet — tool results and interrupted
  sessions are not captured, so stick to 2026.6.11 until that lands.
- `node_bin_dir`: directory put before `PATH` when running `openclaw`. Setup uses the
  same order for its version probes and for `npm`, so a bundled runtime is checked
  as the rollout will use it.
- `OPENCLAW_NODE_VERSION` (env only): Node.js version fetched when `npm` is absent
  (default `24.21.0`, the newest release of the Node 24 LTS line OpenClaw supports).
  The build is picked for the host platform — Linux, macOS and Windows on x64 and
  arm64. Other platforms raise, since nodejs.org publishes no build for them;
  install Node.js yourself and put `npm` on `PATH`.

Runtime compatibility is validated before reuse: a system `npm` is only used when
its `node` satisfies the `engines.node` range of the requested OpenClaw release
(`setup_openclaw.OPENCLAW_ENGINES_NODE` for pinned releases, `npm view` for
others), and a cached local toolchain is only
reused when it reports exactly the requested Node version. Incompatible runtimes
are replaced, not bypassed silently.

The host installer settings above apply only to local CLI calls, with installation deferred
until the first call rather than startup. Concurrent callers share one background-thread
attempt; it does not block the event loop, and failed installs can be retried.
Sandbox sessions use their separate in-sandbox installer and the exact
configured `openclaw_version`; host environment overrides do not change that runtime.

`configs/openclaw_agent.yaml` supports both paths: an agent session selects sandbox
execution; no agent session selects the local CLI. Invalid or closed sessions are
rejected, never retried on the host.
Local `/run` requires a Resources binding; direct `/v1/responses` does not.


## EnvironmentServer sandbox sessions

The sandbox path runs OpenClaw and its file/process tools inside the task sandbox created by
Resources. The agent server borrows `SandboxAccess`, installs the pinned runtime, launches one
invocation in `SandboxAccess.workdir`, and disconnects after confirmed cleanup. Resources retains
ownership of task preparation, verification, and sandbox destruction.

Load the independent benchmark, harness, and generic EnvironmentServer definitions. For example,
with an already prepared SWE-Pro JSONL and a reachable policy model, create a run config:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

# Route existing prepared flat rows through sandbox session orchestration.
environment_routing_mode: legacy
environment_server_name: single_agent_turn_legacy

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server: {type: resources_servers, name: swebench_pro_resources_server}
      agent_server: {type: responses_api_agents, name: openclaw_agent}
```

Save it as `run.yaml`, then run:

```bash
gym env start --config run.yaml --model-type openai_model

NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING=1 \
  gym eval run --no-serve --config run.yaml --model-type openai_model \
  --agent openclaw_agent \
  --input /path/to/prepared-swe-pro.jsonl --output outputs/openclaw-rollout.jsonl --limit 1
```

Supply the sandbox provider config through `env.yaml` or an additional `--config` file, and ensure
`policy_model_name`, `policy_base_url`, and `policy_api_key` match the reachable model endpoint.
Prepared rows keep benchmark task data and prompts; `task_source`, when present, must name the
selected Resources server. `--agent` selects the harness for existing rows. No extra standalone
materialization script is required.

SWE-Pro currently has a legacy `allowed_agents` list that excludes OpenClaw. The explicit
`NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING=1` opt-in permits this smoke without replacing benchmark or
verifier settings. Use the environment form with `--no-serve`: the current collector reloads its
server config from the head server and can lose the CLI-only `--allow-unsupported-pairing` flag. It does not establish compatibility with other benchmarks. Switch compatible components
by changing their independent config paths and the two EnvironmentServer references; keep
benchmark data, preparation, and verifier settings with Resources. There is no combined
benchmark/harness preset.

Submit episodes to **EnvironmentServer `/run`**. It seeds Resources, opens the agent session,
calls the rollout-scoped Responses endpoint, closes the agent, verifies, and closes Resources.
The model server address must be reachable from inside the task sandbox. Its `/ng-rollout/.../v1`
route is embedded in OpenClaw's Chat Completions provider configuration to retain model-call linkage.

The sandbox adapter sets OpenClaw's provider `timeoutSeconds` to `model_timeout_seconds` (600 seconds
by default). This raises the upstream 120-second idle watchdog so buffered reasoning responses can
finish. For example, set `openclaw_agent.responses_api_agents.openclaw_agent.model_timeout_seconds`
to `1200` for a 20-minute model-call limit. Increase the separate `timeout` and EnvironmentServer episode
deadline when a task needs more total time; increasing those alone does not raise the model-idle limit.

The installer supports Linux glibc on x86_64/aarch64 and musl on x86_64, with Python 3.8 or later for the
supervisor. Session setup installs missing Python and Bash with apt-get or apk
when running as root; otherwise preinstall Python 3.8+ and Bash. Existing tools
are preserved. Node 22.19.0 and OpenClaw are installed under
`/tmp/nemo-gym-openclaw-node-22.19.0-<version>`. Node archives are checked against upstream SHA-256
sums; the installed package version is verified before caching and when reusing a cached runtime.
A version-scoped `flock` serializes runtime installation and cache validation across session setup
attempts. Missing `flock`, `curl`, CA certificates, `tar`, `gzip`, `sha256sum`, and `awk` are installed with `apt-get` or `apk` only when
running as root. Other images receive an actionable prerequisite error. Installation errors include
the failed command, exit status, provider error, stdout, and stderr. Supporting these architectures does not establish
validation on every task image; record the actual image digest and runtime versions with each run.

The runner uses ordinary sandbox `exec`; no PTY/session API is required. Its internal deadline
leaves time for descendant cleanup before the provider deadline. Session close requests a stop and
waits for a cleanup receipt before detaching. A disconnected HTTP waiter leaves the shared invocation running. A missing or failed receipt blocks verification.
For old musl images, a checksum-pinned C++ runtime is extracted privately and used only by the
private Node binary; task Node/Python and the global library search path are not replaced.

Per-session HOME, config, caches, prompt, transcript, and supervisor output are isolated under
`/tmp/nemo-gym-openclaw-sessions/`. OpenClaw reads the generated config directly without onboarding
or creating bootstrap files in the task repository. The task's PATH and dependencies are preserved.
The sandbox adapter enables `read`, `write`, `edit`, `exec`, and `process`; subagent/channel/plugin
execution and required external HTTP/MCP tools are unsupported. OpenClaw's internal `gateway`
exec host refers to the embedded CLI process inside the borrowed sandbox.

Sandbox request support is deliberately explicit:

- One activation per session; one worker per agent server. Concurrent sessions are independent.
  Identical activation requests join the running task or replay its result/error. A different request
  is rejected with HTTP 409. Session close owns cancellation; waiter cancellation does not stop the harness.
- EnvironmentServer assigns the session ID. Repeating the full seed request is idempotent, even
  without the original cookie; reusing its ID for different inputs is rejected. Close accepts the
  explicit ID and episode without a cookie, including after a lost seed response. Closing an
  unknown ID prevents a delayed seed from creating it. Filesystem paths use independent random IDs.
- A text user prompt, optionally preceded by one system message. Text-part arrays are accepted.
  Configured `system_prompt`, request `instructions`, and the optional system message are joined
  and prepended to OpenClaw's user prompt. The harness retains its own system prompt.
- The request `model`, when supplied, must match the configured model.
- `max_output_tokens`, `temperature`, `top_p`, reasoning overrides, tool-selection controls, output
  schemas, history replay, request `metadata`, and other unsupported Responses options are rejected before activation.
  This includes `metadata.extra_body.seed` added by `num_repeats_add_seed`; disable that option
  for sandbox sessions. Request `metadata.chat_template_kwargs` is also rejected; configure it
  on the Gym model server instead.
  The sandbox config also rejects `max_output_tokens`: model metadata does not prove an effective
  inference limit. Configure sampling and per-call limits on the Gym model server and inspect the
  effective captured requests. An episode-wide token budget is not implemented.
- Custom command, environment, OpenClaw config, Node path, extra arguments, and agent-ID overrides
  are rejected for sandbox sessions. Existing callers without an agent-session cookie retain the
  local CLI behavior and configuration. Both paths now include request `instructions` in the
  same configured-system, request-instructions, input-system order.

Successful and limit-truncated Responses retain reasoning, tool calls/results, and final or partial text.
Runtime, provider, and upstream model failures raise an error before verification, with transcript
evidence and failed status retained in the observations returned by close.
Usage sums observed assistant model calls, including cache reads/writes and failed final calls.
Transcript rewrites retaining the same response ID and message count once; synthetic CLI summary
messages carrying cumulative usage do not count again or override the model's terminal status.
Conflicting messages with the same response ID are excluded with an accounting gap. Records without
response IDs count separately with an identity gap, since their duplication cannot be established.
Auxiliary model calls, such as compaction, are absent from the transcript. A coverage gap is always
reported because interrupted compaction need not leave an event. Totals therefore need not equal
all backend calls. Missing per-call usage and
unavailable reasoning-token details are also reported as observation gaps; the full branch history is retained.
An explicit wall or model-output limit returns partial output as `incomplete`; upstream/model errors
raise HTTP 502 even when a useful patch exists. Do not infer rollout success from reward alone. Close returns the captured observations,
including salvaged transcript evidence after cancellation.

A Linux child subreaper supervises each activation and reaps detached/background descendants before
writing an explicit cleanup receipt. Close is serialized and checks the receipt, process handle,
adapter-owned file removal, and disconnect before succeeding. Unknown launch outcomes and missing
or negative cleanup evidence fail closed and block verification. Successful close receipts remain
retryable for `session_close_retry_window_seconds` after cleanup (300 seconds by default); traffic
and retries do not shorten or extend the window. Stale cookies continue to reject activations after
receipt expiry and never enter the host CLI path.

The shared `SimpleResponsesAPIAgent` lifecycle owns session identity, serialization, and close
receipts. `session_lifetime_seconds` has been removed: the episode owner and provider TTL govern
abandoned-session recovery; there is no adapter timer cancelling active sessions. Setup failures
whose cleanup fails retain cleanup-only state through `AgentSessionSetupError`.
`timeout`, `model_timeout_seconds`, `sandbox_install_timeout_seconds`, and `session_close_timeout_seconds` must be positive
and finite. The shared `session_close_retry_window_seconds` bounds close receipts and tombstones.

Session state is process-local. Keep requests on one worker. Failed cleanup sessions are retained
until owner/provider recovery and agent restart; they are never relabeled as successfully closed.
Cached runtime files remain until Resources destroys the sandbox. Supervisor cleanup prevents
verification races; it is not a security boundary against hostile task code.

This path is eval-only. Working inference does not establish training token-ID/logprob support.
A tool-and-verifier smoke is required for the actual image/model configuration before review;
unit tests alone do not establish runtime compatibility or benchmark accuracy. Process/memory/setup
and close-latency overhead have not been benchmarked at scale.

Without `sandbox_access`, configure `sandbox_provider` and `sandbox_config`
(`SandboxSpec` fields such as `image`, `workdir`, and `ttl_s`) on the agent.
It creates a sandbox, runs there, and destroys it on close; the default workdir is
`/app`. A supplied access always wins, including its workdir; connection failure
never triggers a replacement or host execution. With neither access nor a usable
provider, setup fails. Verifiers that inspect task files must keep using a
Resources-owned sandbox, since agent-owned sandboxes are gone before verification.
