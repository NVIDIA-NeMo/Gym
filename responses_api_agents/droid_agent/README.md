# Droid Agent

Runs [Factory Droid](https://factory.ai) (`droid exec`) as a NeMo Gym agent server. Droid runs
inside the task sandbox of an Environment Server agent session, with its own shell and file tools,
and sends every model call to Gym's Model Server through Droid's bring-your-own-key (BYOK)
custom-model support. It needs no Factory account, login or API key.

- Harness version: Droid **1.1.0** (`droid_version`), installed from the npm platform packages
  `@factory/cli-linux-x64`, `@factory/cli-linux-x64-baseline` (x86-64 without AVX2) and
  `@factory/cli-linux-arm64`.
- Execution: sandbox only. A borrowed sandbox from the benchmark's Resources Server, or an
  agent-owned sandbox from `sandbox_provider`. There is no host path and no compatibility `/run`.
- Capability level: Evaluation. The close bundle carries the invocation, the conversation, tool
  timing and outcome, the sandbox record and the run's usage; see [Observations](#observations).
- Model API: Chat Completions (streaming), through the rollout-prefixed Gym Model Server URL.
- Tool grants: none. Required Resources tool grants are refused at seed; keep
  `resources_tool_transports: []`.

## Configure and run

[configs/droid_agent.yaml](configs/droid_agent.yaml) is the harness definition. Compose it with a
benchmark and a `single_agent_turn` Environment Server. For Terminal-Bench 2.1, save this as
`run.yaml`:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/droid_agent/configs/droid_agent.yaml
  - environment_servers/single_agent_turn/configs/single_agent_turn.yaml

environment_routing_mode: taskset
environment_server_routes:
  terminal_bench_2_1: single_agent_turn

single_agent_turn:
  environment_servers:
    single_agent_turn:
      resources_server:
        name: terminal_bench_2_1_resources_server
      agent_server:
        name: droid_agent
      resources_tool_transports: []

droid_agent:
  responses_api_agents:
    droid_agent:
      context_window: 131072
      max_output_tokens: 8192
      timeout: 1200
```

Supply the `policy_model` Model Server, `policy_model_name` and the `sandbox` provider in a
separate model and provider config, as the
[Terminal-Bench 2.1 README](../../benchmarks/terminal_bench_2_1/README.md) shows. Droid replays its
assistant history with an empty `reasoning_content` and the model's `<think>` text in `content`; with
`uses_reasoning_parser: true` the vLLM Model Server then sends both `reasoning` and
`reasoning_content`, which NVIDIA NIM rejects (`400 duplicate field reasoning_content`). For NIM, set
`reasoning_field: reasoning_content` on the `policy_model` server. The sandbox must
reach the Model Server: set `++use_absolute_ip=true` when the Model Server would otherwise
advertise a loopback address. Seed refuses a loopback Model Server URL for a remote provider.
Terminal-Bench 2.1 does not list `droid_agent` in its `allowed_agents` yet; until it does, start
the servers with `--allow-unsupported-pairing`.

## Runtime installation

Seed probes the sandbox's architecture, libc and AVX2 support, then downloads the matching npm
package on the **Gym host**, into the owner-only `~/.cache/nemo_gym/droid/<version>/` cache. The
package must match its npm `sha512` integrity. For 1.1.0 the integrity of each package is pinned in
[runtime.py](runtime.py); another version's integrity comes from the npm registry's metadata. The
package (about 150 MB) is uploaded to the sandbox in 1 MiB pieces, and
[install_droid_runtime.sh](install_droid_runtime.sh) checks its SHA-256, unpacks the `droid`
binary into `/tmp/nemo-gym-droid-runtime-<version>`, checks that it reports the pinned version, and
records the binary's SHA-256. Later seeds in the same sandbox reuse a runtime whose binary is
unchanged. A lock serializes concurrent installers. The sandbox needs no network for installation.

The task image needs bash, coreutils (`sha256sum`, `timeout`), `tar` and `gzip`, and glibc: Droid
publishes no musl build, so Alpine images are refused at seed. A root image without bash gets it
from `apt-get` or `apk`. The image needs no Python or Node.

## Requests and settings

A request is one user text message, optionally preceded by a system message. The prompt is passed
with `--file`. The configured `system_prompt` and the request's system message are appended to
Droid's own system prompt with `--append-system-prompt-file`. `metadata.agent_timeout_sec` can lower
the `timeout` ceiling for one task. Every other request field (sampling, `max_output_tokens`,
`instructions`, `tools`, other metadata, history) is refused with HTTP 422 instead of being
dropped. Keep sampling settings on the Model Server: Droid 1.1.0 sends `temperature: 1` and
`max_tokens: <max_output_tokens>` on every call.

Each activation writes a runtime settings file (`--settings`) with one BYOK model:

| Droid setting | Value |
| --- | --- |
| `model` | `model` (default `${policy_model_name}`), selected with `--model custom:<model>` |
| `provider` | `generic-chat-completion-api` |
| `baseUrl` | the Model Server URL with this rollout's `/ng-rollout/<capture key>` prefix and `/v1` |
| `apiKey` | a placeholder; the provider credential stays on the Model Server |
| `maxOutputTokens` | `max_output_tokens` |
| `maxContextLimit` | `context_window`, when set; Droid compacts the context against it |
| `extraHeaders` | `x-session-id: <capture key>` |

Droid sends the URL and header on every model call of the run, including the calls of subagents it
starts with its Task tool; context compaction uses the same URL. Model-call capture can then assign
each call, a failed one too, to the invocation. Droid runs with `--skip-permissions-unsafe`: the task sandbox is the isolation boundary.

## Files, environment and network

- **State.** `FACTORY_HOME_OVERRIDE` moves all of Droid's state (sessions, logs, caches, its
  self-kill guard) under the private session directory `/tmp/nemo-gym-droid-sessions/<id>/home`,
  which close removes. Droid writes nothing to the task repository or the task's home.
- **Tool environment.** Tool commands keep the sandbox's environment, including the task's `HOME`,
  `XDG_*` directories and working directory. Droid adds its own variables to them, among them
  `FACTORY_HOME_OVERRIDE`, `FACTORY_SESSION_ID`, `NODE_EXTRA_CA_CERTS` (a certificate bundle in its
  cache), `GIT_TERMINAL_PROMPT=0`, `GIT_EDITOR=true`, `DEBIAN_FRONTEND=noninteractive` and `NO_COLOR`,
  and it puts `pkill` and `killall` guards first on `PATH` so a tool cannot kill Droid itself.
- **Network.** With BYOK, model calls go only to the configured `baseUrl`. Droid also contacts
  Factory's servers for feature flags, billing limits, usage records and telemetry; without a
  Factory login these calls fail and the run continues. The adapter sets `FACTORY_AIRGAP_ENABLED=1`,
  which stops telemetry; Droid 1.1.0 still attempts one automation-sync request to
  `api.factory.ai`, which fails without credentials. A run with no network except the Model Server
  completes.

## Results and lifecycle

Droid's `--output-format stream-json` events become the response: assistant text as messages,
reasoning as reasoning items, and every tool call and result, in order. A tool result that is not
text is serialized as JSON. Subagent reports that Droid injects as user messages stay in the
invocation's conversation, not in the response output, and a gap records that.

| Run ends with | Response status |
| --- | --- |
| a `completion` event and exit 0 | `completed` |
| the time limit (or a rollout stop) | `incomplete` |
| exit 0 without a `completion` event | `incomplete` |
| an `error` event, or a nonzero exit | `failed`, with `ng_agent_outcome` |

An error event that starts with an HTTP status (Droid 1.1.0 reports a failed model request as
`404 {...}`) is classified as `model_request_error` with that status; connection failures as
`model_transport_error`; anything else as `unknown`. The Environment Server's outcome policy then
decides whether to verify or retry.

Session identity, the single activation, retries, close and its retry window follow the shared
agent-session contract. `SandboxSession` runs Droid through the shared `harness_runner.sh` under the
process supervisor, stops it at the deadline, on a rollout stop or on close, and collects stdout,
the stderr tail and the runner identity after termination is confirmed. Close disconnects a
borrowed sandbox and stops an owned one.

## Observations

The close bundle (`source: droid`) has:

- an `AgentInvocation` with the capture key as `invocation_id`, Droid's session id, status, error
  type, duration and the conversation (system and user input, then the run's items);
- a `ToolCallObservation` per tool call, timed by Droid's event timestamps. A tool reported as an
  error, or an `Execute` command whose result ends with a nonzero `[Process exited with code N]`,
  is `failed`;
- a `SandboxObservation` with the sandbox id and the harness exit code; tool records carry the
  same sandbox id.

The response usage is Droid's run total from its `completion` event, which counts every call,
compaction and subagents included. Droid counts cache reads and writes apart from `input_tokens`, as
Anthropic's usage does, so the response's input total adds them back, and `cached_tokens` is the
cache reads. Droid 1.1.0 reports zero thinking tokens for a Chat Completions model that reasons, so
reasoning tokens are reported as 0 with a `reasoning_token_usage_unavailable` gap; the captured
model calls carry the provider's count. Without the `completion` event the usage is unknown and the
bundle records `model_call_usage_unavailable`. Droid's stream-json output does not name the served
response of each call, so the invocation lists no `ModelCallRef`s; model-call capture joins calls
to it through the `x-session-id` header. Other gaps: `subagent_hierarchy_unavailable` when a Task
tool call started a subagent (its own transcript stays in Droid's private session directory),
`injected_user_messages_omitted`, `tool_timing_unavailable`, `agent_stdout_unparseable` and
`runtime_info_unavailable`.

## Limits

- In some container images, Droid kills its own foreground shell commands. On Docker 28.1.1, every
  foreground `Execute` call in the Terminal-Bench 2.1 images that ship Python
  (log-summary-date-ranges, openssl-selfsigned-cert, cancel-async-tasks), and on Enroot 3.4.1 in
  `python:3.13-slim`, returned `Error: Command terminated by signal: SIGKILL`, whatever the command;
  background `Execute` calls ran. Foreground calls ran normally in configure-git-webserver (Ubuntu
  24.04 without Python) and on a bare host. Droid 0.251.0, 1.0.0, 1.1.0 and 1.2.0 behave the same;
  running as non-root, with a TTY or without seccomp did not change it. Models can still finish tasks
  by running commands in the background, but expect lower scores where this happens. The adapter
  reports these calls as failed tools with Droid's message as the output.
- Droid is distributed under Factory's proprietary terms (`UNLICENSED` on npm). The adapter
  downloads the published binary at seed; it vendors nothing.
- Glibc Linux sandboxes only (x86-64 or arm64).
- No MCP or direct HTTP tool grants, no host execution, no `/run`.
- Context compaction is not reported as a `ContextCompactionObservation`: Droid's stream-json
  output has no compaction event.
- Training-token capture is not validated for this adapter.
