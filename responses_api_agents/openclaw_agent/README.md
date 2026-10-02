# OpenClaw Agent

Runs OpenClaw CLI (`openclaw agent --local --json`).
OpenClaw runs its own tools internally.
Resources server is used for verifier.

Minimal, meant to be extended, and currently eval-only. 

## Quick start

OpenClaw is auto-installed on the first local CLI invocation. Starting an agent server for native
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

## Model id

OpenClaw drops the leading `<provider>/` to form the upstream id,
so we include an extra prefix, such as for `nvidia/...` ids:

```yaml
model: nvinf/nvidia/meta/llama-3.3-70b-instruct
openclaw_config:
  models:
    providers:
      nvinf:
        api: openai-completions
        baseUrl: ${policy_base_url}
        apiKey: ${policy_api_key}
        models:
        - {id: nvidia/meta/llama-3.3-70b-instruct, name: nvidia/meta/llama-3.3-70b-instruct, api: openai-completions}
```

Alternatively, set `model_server` to a Gym model server and set `model` to its served model id. The
agent creates the OpenClaw provider entry automatically. Without `model_server`, the existing
provider configuration is unchanged.

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
- `extra_args`: extra flags appended to `openclaw agent`
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `openclaw_config`: deep-merged into the generated `openclaw.json`
- `openclaw_version`: exact npm version to install (required; native validation uses `2026.6.11`)

See `configs/openclaw_agent.yaml`.


## Native EnvironmentServer sessions

The native path runs OpenClaw and its file/process tools inside the task sandbox created by
Resources. The agent server borrows `SandboxAccess`, installs the pinned runtime, launches one
invocation in `SandboxAccess.workdir`, and disconnects after confirmed cleanup. Resources retains
ownership of task preparation, verification, and sandbox destruction.

Load the independent benchmark, harness, and generic EnvironmentServer definitions. For example,
with an already prepared SWE-Pro JSONL and a reachable policy model, create a run config:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/openclaw_agent/configs/openclaw_agent_native.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

# Route existing prepared flat rows through native session orchestration.
environment_routing_mode: legacy
environment_server_name: single_agent_turn_legacy

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server: {type: resources_servers, name: swebench_pro_resources_server}
      agent_server: {type: responses_api_agents, name: openclaw_agent_native}
```

Save it as `native-run.yaml`, then run:

```bash
gym env start --config native-run.yaml --model-type openai_model

NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING=1 \
  gym eval run --no-serve --config native-run.yaml --model-type openai_model \
  --agent openclaw_agent_native \
  --input /path/to/prepared-swe-pro.jsonl --output outputs/openclaw-native.jsonl --limit 1
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

The installer supports Linux glibc on x86_64/aarch64 and musl on x86_64, with Python 3.8 or later for the
supervisor. Node 22.19.0 and OpenClaw are installed under
`/tmp/nemo-gym-openclaw-node-22.19.0-<version>`. Node archives are checked against upstream SHA-256
sums; the installed package version is verified before caching and when reusing a cached runtime.
A version-scoped `flock` serializes runtime installation and cache validation across session setup
attempts. Missing `flock`, `curl`, CA certificates, `tar`, `gzip`, `sha256sum`, and `awk` are installed with `apt-get` or `apk` only when
running as root. Other images receive an actionable prerequisite error. Installation errors include
the failed command, exit status, and stderr. Supporting these architectures does not establish
validation on every task image; record the actual image digest and runtime versions with each run.

The runner uses ordinary sandbox `exec`; no PTY/session API is required. Its internal deadline
leaves time for descendant cleanup before the provider deadline. Session close requests a stop and
waits for a cleanup receipt before detaching. A disconnected HTTP waiter leaves the shared invocation running. A missing or failed receipt blocks verification.
For old musl images, a checksum-pinned C++ runtime is extracted privately and used only by the
private Node binary; task Node/Python and the global library search path are not replaced.

Per-session HOME, config, caches, prompt, transcript, and supervisor output are isolated under
`/tmp/nemo-gym-openclaw-sessions/`. OpenClaw reads the generated config directly without onboarding
or creating bootstrap files in the task repository. The task's PATH and dependencies are preserved.
The native adapter enables `read`, `write`, `edit`, `exec`, and `process`; subagent/channel/plugin
execution and required external HTTP/MCP tools are unsupported. OpenClaw's internal `gateway`
exec host refers to the embedded CLI process inside the borrowed sandbox.

Native request support is deliberately explicit:

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
  schemas, history replay, and other unsupported Responses options are rejected before activation.
  The native config also rejects `max_output_tokens`: model metadata does not prove an effective
  inference limit. Configure sampling and per-call limits on the Gym model server and inspect the
  effective captured requests. An episode-wide token budget is not implemented.
- Custom command, environment, OpenClaw config, Node path, extra arguments, and agent-ID overrides
  are rejected for native sessions. Existing callers without an agent-session cookie retain the
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
`timeout`, `sandbox_install_timeout_seconds`, and `session_close_timeout_seconds` must be positive
and finite. The shared `session_close_retry_window_seconds` bounds close receipts and tombstones.

Session state is process-local. Keep requests on one worker. Failed cleanup sessions are retained
until owner/provider recovery and agent restart; they are never relabeled as successfully closed.
Cached runtime files remain until Resources destroys the sandbox. Supervisor cleanup prevents
verification races; it is not a security boundary against hostile task code.

This path is eval-only. Working inference does not establish training token-ID/logprob support.
A tool-and-verifier smoke is required for the actual image/model configuration before review;
unit tests alone do not establish runtime compatibility or benchmark accuracy. Process/memory/setup
and close-latency overhead have not been benchmarked at scale.
