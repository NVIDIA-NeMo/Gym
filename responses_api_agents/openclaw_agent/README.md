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

Bind the agent, Resources, model server, and sandbox provider in your environment config. For
example, the following agent/environment blocks refer to an existing Resources server named
`task_resources` and a Gym model server named `policy`:

```yaml
openclaw_native:
  responses_api_agents:
    openclaw_agent:
      entrypoint: app.py
      num_workers: 1
      openclaw_version: 2026.6.11
      model_server: {type: responses_api_models, name: policy}
      model: your-served-model-id
      timeout: 900
      sandbox_install_timeout_seconds: 900
      session_close_timeout_seconds: 60
      session_close_retry_window_seconds: 300

native_environment:
  environment_servers:
    single_agent:
      entrypoint: app.py
      resources_server: {type: resources_servers, name: task_resources}
      agent_server: {type: responses_api_agents, name: openclaw_native}
```

Submit episodes to **EnvironmentServer `/run`**. It seeds Resources, opens the agent session,
calls the rollout-scoped Responses endpoint, closes the agent, verifies, and closes Resources.
The model server address must be reachable from inside the task sandbox. Its `/ng-rollout/.../v1`
route is embedded in OpenClaw's Chat Completions provider configuration to retain model-call linkage.

The installer supports Linux glibc on x86_64/aarch64 and requires Python 3.9 or later for the
supervisor. Node 22.19.0 and OpenClaw are installed under
`/tmp/nemo-gym-openclaw-node-22.19.0-<version>`. Node archives are checked against upstream SHA-256
sums; the installed package version is verified before caching and when reusing a cached runtime.
A version-scoped `flock` serializes runtime installation and cache validation across session setup
attempts. Missing `flock`, `curl`, CA certificates, `tar`, `xz`, `sha256sum`, and `awk` are installed with `apt-get` only when
running as root. Other images receive an actionable prerequisite error. Installation errors include
the failed command, exit status, and stderr. Supporting these architectures does not establish
validation on every task image; record the actual image digest and runtime versions with each run.

Per-session HOME, config, caches, prompt, transcript, and supervisor output are isolated under
`/tmp/nemo-gym-openclaw-sessions/`. OpenClaw reads the generated config directly without onboarding
or creating bootstrap files in the task repository. The task's PATH and dependencies are preserved.
The native adapter enables `read`, `write`, `edit`, `exec`, and `process`; subagent/channel/plugin
execution and required external HTTP/MCP tools are unsupported. OpenClaw's internal `gateway`
exec host refers to the embedded CLI process inside the borrowed sandbox.

Native request support is deliberately explicit:

- One activation per session; one worker per agent server. Concurrent sessions are independent.
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
  local CLI behavior and configuration.

Responses retain reasoning, tool calls/results, final or partial text, and terminal failure status.
Usage sums observed assistant model calls, including cache reads/writes and failed final calls.
Transcript rewrites retaining the same response ID and message count once; synthetic CLI summary
messages carrying cumulative usage do not count again or override the model's terminal status.
Conflicting messages with the same response ID are excluded with an accounting gap. Records without
response IDs count separately with an identity gap, since their duplication cannot be established.
Auxiliary model calls, such as compaction, are absent from the transcript. A coverage gap is always
reported because interrupted compaction need not leave an event. Totals therefore need not equal
all backend calls. Missing per-call usage and
unavailable reasoning-token details are also reported as observation gaps; the full branch history is retained.
A timeout returns partial output as `incomplete`; model errors remain `failed` even when a useful
patch exists. Do not infer rollout success from reward alone. Close returns the captured observations,
including salvaged transcript evidence after cancellation.

A Linux child subreaper supervises each activation and reaps detached/background descendants before
writing an explicit cleanup receipt. Close is serialized and checks the receipt, process handle,
adapter-owned file removal, and disconnect before succeeding. Unknown launch outcomes and missing
or negative cleanup evidence fail closed and block verification. Successful close receipts remain
retryable for `session_close_retry_window_seconds` after cleanup (300 seconds by default); traffic
and retries do not shorten or extend the window. Stale cookies continue to reject activations after
receipt expiry and never enter the host CLI path.

Session state is process-local. Keep requests on one worker. Failed cleanup sessions are retained
until owner/provider recovery and agent restart; they are never relabeled as successfully closed.
Cached runtime files remain until Resources destroys the sandbox. Supervisor cleanup prevents
verification races; it is not a security boundary against hostile task code.

This path is eval-only. Working inference does not establish training token-ID/logprob support.
A tool-and-verifier smoke is required for the actual image/model configuration before review;
unit tests alone do not establish runtime compatibility or benchmark accuracy. Process/memory/setup
and close-latency overhead have not been benchmarked at scale.
