# Cline Agent

Runs the [Cline](https://cline.bot) CLI headlessly (`cline --json`). Cline runs its own tools
internally, and its newline-delimited JSON event stream is parsed into Gym format and verified by
the resources server.

Capability level: **Evaluation**. Agent sessions also return an invocation, tool timings and a
sandbox record (see [Observations](#observations)); token IDs and logprobs are not wired up.

| Request | Where Cline runs | Config |
|---|---|---|
| Agent session seeded with `SandboxAccess` (recommended) | The task sandbox the Resources Server supplies | [`cline_agent.yaml`](configs/cline_agent.yaml) |
| Agent session without `SandboxAccess`, `sandbox_provider` set | A sandbox the agent creates and stops | `cline_agent.yaml` plus `sandbox_provider` / `sandbox_config` |
| Agent session without either | Nowhere: the seed is refused with HTTP 422 | — |
| `/run`, or `/v1/responses` without a session | The Agent Server host | [`cline_local_agent.yaml`](configs/cline_local_agent.yaml) |

`configs/cline_agent.yaml` is the agent-session config. The host config that file used to hold is
now `configs/cline_local_agent.yaml`, with the same `cline_agent` key; configs that listed the old
path for the host path (as `environments/cline_math` did) point at the new one.

## Agent sessions (task sandbox)

Pair the agent with a benchmark through a `single_agent_turn` Environment Server. No benchmark
declares Cline yet. The Terminal-Bench 2.1 recipe below is the unsupported development pairing the
adapter's smoke runs used, not a validated pairing: replace the Codex import and settings in the
benchmark README's `run.yaml` with:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/cline_agent/configs/cline_agent.yaml
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
        name: cline_agent
      resources_tool_transports: []
```

Cline is not in the benchmark's `allowed_agents`, so start the servers with
`--allow-unsupported-pairing`; declaring the pairing takes a full evaluation and the benchmark
owner's review. The task sandbox must reach the Gym Model Server; on a shared host,
pass `++use_absolute_ip=true`. Cline replays earlier turns' reasoning, which the vLLM Model Server
forwards as both `reasoning` and `reasoning_content` by default; an endpoint that rejects the pair
(NIM's `minimaxai/minimax-m3.1-flash` answers `duplicate field reasoning_content`) needs
`reasoning_field: reasoning_content` on that Model Server.

**Seed.** The agent connects to the borrowed sandbox (only `direct` connections) and installs the
pinned Cline inside it with [`install_cline_runtime.sh`](install_cline_runtime.sh). Cline 3.0.55 ships a
self-contained glibc binary per platform in `@cline/cli-linux-{x64,arm64}`, so the sandbox needs no
Node. The installer downloads that npm tarball, checks it against the SHA-512 the adapter pins for
the release (`CLINE_RUNTIME_SHA512` in [`sandbox.py`](sandbox.py)), and publishes only the binary
in `/tmp/nemo-gym-cline-runtime-<version>` with one rename; a later session in the same sandbox
reuses it. Requirements: Linux x86_64 or aarch64 with glibc, Bash, and HTTPS access to
`registry.npmjs.org`. When `curl`, `tar`, `gzip`, `sha512sum` or CA certificates are missing, a root
sandbox installs them with `apt-get`, `dnf` or `yum`; otherwise the seed fails with the missing
names. musl images are refused. The seed is also refused (HTTP 422) without an exact
`cline_version` that has pinned checksums, without `model_server` and `model`, for a required tool
grant, and for the host-only settings `command`, `env`, `extra_args`, `repo_dir`,
`openai_api_key` and `openai_base_url`.

Concurrent replacement of a stale runtime uses `flock` with a 60-second wait. Without `flock`, a cold
install and reuse of a verified runtime still work; replacing a stale directory fails without deleting
it. Install `flock` or use a fresh sandbox to recover.

`network_access: inherit` (the default) keeps the provider's network policy. For agent-owned sandboxes,
`network_access: model_only` uses OpenSandbox's policy to deny egress except to the Model Server's host
(all its ports and paths, including co-hosted services). It refuses providers without that policy,
loopback or wildcard model endpoints, and a simultaneous `sandbox_config.provider_options.network_policy`
before allocating a sandbox. Borrowed sandboxes retain the Resources Server's policy and refuse
`model_only`; Cline does not consume external tool grants, so there is no `model_and_tools` mode.

Restricted setup downloads the pinned npm archive on the Agent Server, verifies its SHA-512, and uploads
only the binary. The sandbox checks its transfer digest and version before the same locked publication;
verified runtimes are reused without downloading. The image must already contain Bash, glibc and coreutils;
missing prerequisites are a final seed refusal, with the owned sandbox stopped. No sandbox package manager
or registry access is needed. The Agent Server needs HTTPS access to `registry.npmjs.org`. Under `inherit`,
installation keeps the in-sandbox download and automatic package setup described above.

**Activation.** Cline runs in `SandboxAccess.workdir` under Gym's process supervisor, through the
shared `harness_runner.sh`. Everything Cline writes goes to `/tmp/nemo-gym-cline-sessions/<id>`:
`--config` and `--data-dir` point there, together with the `CLINE_*` state paths. The agent writes
Cline's provider settings there directly — the `openai-compatible` provider with the
rollout-prefixed Model Server URL and an `x-session-id` header carrying the rollout's capture key,
which owns every model attempt, including failed ones — and global settings that turn off
telemetry and self-update. Tool shells keep the task's `HOME` and `XDG_*` directories. Close removes
the session directory, but the runtime in `/tmp/nemo-gym-cline-runtime-<version>`, which later
sessions reuse, any packages the seed installed, and Cline's checkpoint commits (see
[Limitations](#limitations)) stay in the sandbox. The supervisor's deadline is `timeout`, or a
task's `metadata.agent_timeout_sec` below it.

**Close** stops Cline, captures its output, removes the session directory and disconnects; the
Resources Server keeps and later destroys the borrowed sandbox. `POST /rollouts/<id>/stop` ends a
running Cline like its deadline does, where the sandbox provider supports it.

An activation accepts one text user message with an optional system (or developer) message,
`instructions`, the configured model (or the model server's name) and `metadata.agent_timeout_sec`.
Other request fields, such as `temperature` or `max_output_tokens`, are refused with HTTP 422: Cline
has no per-request sampling flags, so set sampling on the model server. The prompt is
`system_prompt`, `instructions` and the system message, then the user message; the host path builds
it the same way but leaves out `instructions`.

| Cline run | Response |
|---|---|
| `finishReason: completed` (Cline exits 0) | `completed` |
| `finishReason: error` | `failed`, with the outcome Cline names: a context overflow is `incomplete` instead, a refused or timed-out connection `model_transport_error`, an empty model response `agent_invalid_output`, anything else `unknown` |
| `finishReason: aborted`, the model's output-token cap, or another limit | `incomplete`, with no typed failure kind; the partial work is verified |
| Deadline after recorded model output | `incomplete`, with invocation `stop_reason: timeout`; the partial work is verified |
| Deadline before any recorded model output | `failed` with `unknown`, invocation `stop_reason: timeout`; retried by default, without claiming a model transport failure |
| No run result (Cline or the runner crashed) | The activation fails (HTTP 500) |

Cline 3.0.55's error events do not retain structured HTTP status codes. Provider HTTP errors, including
400, 429 and 5xx messages, therefore remain `unknown` and are retried by the default outcome policy;
the adapter does not infer HTTP status from text. `agent_outcome_policy.by_failure_kind` can change
that treatment. An explicit rollout stop remains `incomplete` even before model output arrives.

If the sandbox provider also times out waiting after the supervisor confirms its deadline, both facts
survive: the sandbox observation records the harness deadline and the failure report retains the
provider wait error (including an original Gym failure event when available). Retaining this diagnostic
does not turn an otherwise gradable partial response into another episode attempt.

### Observations

The close bundle (`source: cline`) holds one `AgentInvocation` whose ID is the rollout's capture
key, with the run's status, `finishReason` and duration, and the conversation; a
`ToolCallObservation` for each tool call, timed by Cline's own event timestamps, `failed` when the
tool reports an error or an unsuccessful entry; and a `SandboxObservation` with the harness exit
code; the tool calls and the sandbox record carry the sandbox's ID. Model calls are owned through the
`x-session-id` header, not listed on the invocation. Turns, subagent hierarchy (`spawn_agent` and team
tools) and compactions are not recorded.

A tool call still running when the deadline or a stop ends Cline is recorded as `timeout` or
`cancelled`, and the observed conversation gets an output saying it returned none; the response keeps
only what Cline emitted.

Response usage is Cline's final `run_result` total; a run stopped before it reports usage as unknown,
with a `model_call_usage_unavailable` gap, because Cline's running totals can lag calls the Model Server
already answered. For the `openai-compatible` provider the input count already includes cache reads,
which are reported as `cached_tokens` and not added again; Cline 3.0.55 reports no reasoning count for
it, so `reasoning_tokens` stays unknown, with a `reasoning_token_usage_unavailable` gap. The host path still adds cache reads to the input count (see
Limitations).

## Host path

Cline must be on PATH; the first host call installs it (or `npm install -g cline`). Concurrent first
calls share one installation, a failed installation fails those calls, and the next call retries it.
Agent sessions never install Cline on the host. Set `policy_base_url`, `policy_api_key`, and
`policy_model_name` in `env.yaml`; the model server started by `--model-type` serves that backend and
Cline calls the model server.

```bash
gym env start \
  --config environments/cline_math/config.yaml \
  --model-type openai_model

gym eval run --no-serve --agent cline_math_agent \
  --input environments/cline_math/data/example.jsonl \
  --output cline_rollout.jsonl
```

Per request the agent creates an isolated run dir, runs one `cline --json --auto-approve true`, and
parses the streamed JSON events for the trajectory. The run dir holds a per-request Cline data
directory (`CLINE_DATA_DIR` and the settings, session, db, and team paths all point inside it), so
concurrent rollouts share no provider settings or session state and the user's `~/.cline` is never
touched or read. `CLINE_SESSION_BACKEND_MODE=local` keeps each run in its own process rather than a
shared hub daemon. The whole run dir is removed afterwards.

Both the host path (`/v1/responses` and `/run`) and agent sessions refuse non-null request
`max_output_tokens` with HTTP 422. Rejection precedes host CLI setup or session activation binding;
`/run` also refuses it before seeding Resources. Cline cannot enforce a shared output budget across its
model calls. Remove this field from task rows or activation requests and configure a per-call cap on
the Model Server instead, such as
`policy_model.responses_api_models.vllm_model.extra_body.max_tokens`. A corrected request can run normally.

## SWE-bench

`anyswe_agent` runs this harness inside a SWE task image, extracts the repository patch, and grades
it — see [`anyswe_cline.yaml`](../anyswe_agent/configs/anyswe_cline.yaml) and the
[anyswe_agent README](../anyswe_agent/README.md):

```bash
export ANYSWE_CONTAINER_FORMATTER='registry.example.com/anyswe/swebench:{instance_id}'
python3 responses_api_agents/anyswe_agent/prepare.py --limit 5

gym env start \
  --config responses_api_agents/anyswe_agent/configs/anyswe_cline.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model

gym eval run --no-serve --agent anyswe_cline \
  --input responses_api_agents/anyswe_agent/data/swebench_verified.jsonl \
  --output results/anyswe_cline_rollouts.jsonl --limit 5
```

That config sets `repo_dir: /testbed` so Cline edits the checkout anyswe takes the patch from.
`setup_scripts/cline_agent_deps.sh` builds the portable runtime (`agent_runtime_source: auto`) when
the CLI is not already baked into the task image.

## Model server

With `model_server` set (the shipped default), Cline's model calls go to that Gym model server
rather than to a provider directly. That is what makes requests and responses show up in Gym's
capture, and it means one config runs against vLLM, OpenAI, or an inference provider by swapping
`--model-type`.

Cline reads its provider, key, model, and base URL from a provider settings file rather than from
run flags — `-k` overrides only the key, and no flag sets a base URL — so on this path the agent
runs one `cline auth openai-compatible --baseurl <model server> --modelid <model> --data-dir <run
dir>` before the task, writing those settings into the run's own data dir. `model` is therefore
required here, and is the bare name the model server serves:

```yaml
model_server: {type: responses_api_models, name: policy_model}
model: ${policy_model_name}
```

The provider is forced to `openai-compatible` on this path: `cline auth` accepts a base URL only for
the OpenAI and OpenAI-compatible providers. The API key written to the run's settings is the literal
`EMPTY` — Gym model servers do not check it, and the real credential (if any) lives on the model
server. During a `/run` the base URL carries the per-rollout `/ng-rollout/<id>` prefix, so captured
model calls are attributable to the rollout that made them.

### Calling a provider directly

Set `model_server: null` and name the provider yourself. Cline then uses whatever that provider is
already authenticated with (`cline auth` outside Gym, or `openai_base_url` / `openai_api_key` for
the OpenAI-compatible one), and no model calls are captured:

```yaml
model_server: null
provider: anthropic
model: claude-sonnet-4-5
```

## Config fields

- `model_server`: Gym model server Cline calls; `null` to use an already-configured provider
- `concurrency`: max simultaneous `run()` calls, and of Cline runs in agent sessions
- `command`: the Cline command, split on spaces so a multi-word launcher works (e.g. `npx cline`)
- `model`: the model id; required with a model server, `null` otherwise leaves the provider's own
  selection (`-m` is omitted)
- `provider`: provider id for `-P`; ignored with a model server (forced to `openai-compatible`)
- `openai_api_key` / `openai_base_url`: passed to the subprocess as `OPENAI_API_KEY` /
  `OPENAI_BASE_URL`; a model server takes precedence over both
- `env`: extra env vars for the subprocess
- `workspace_root`: where per-request run dirs are created and deleted
- `repo_dir`: optional persistent project dir to run in (default: ephemeral per-request dir)
- `system_prompt`: prepended to the user message
- `system_prompt_override`: passed as `-s`, which **replaces** Cline's built-in system prompt rather
  than adding to it, dropping the instructions its tools are described by. Prefer `system_prompt`
- `thinking`: `--thinking` level (`none|low|medium|high|xhigh`); `null` leaves the provider default
- `compaction`: `--compaction` mode (`agentic|basic|off`); `null` leaves Cline's default
- `retries`: `--retries`, max consecutive mistakes before Cline halts
- `timeout`: seconds for one Cline run. The host path kills the subprocess then and also passes
  `--timeout`, which Cline 3.0.55 does not apply to a `--json` run; in a sandbox the process
  supervisor enforces it
- `setup_timeout`: seconds for the one-off `cline auth` call
- `sandbox_provider` / `sandbox_config`: agent sessions only, the provider block and `SandboxSpec`
  fields for an agent-owned sandbox when the Resources Server supplies none
- `sandbox_install_timeout_seconds`: budget for preparing the session and installing Cline in the
  sandbox
- `extra_args`: extra flags appended to the `cline` invocation
- `command_permissions`: JSON policy passed as `CLINE_COMMAND_PERMISSIONS`, restricting the shell
  commands Cline may run (e.g. `{allow: ["python3 *"], deny: ["sudo *"]}`); empty means unrestricted
- `cline_version`: `cline` npm version installed on a clean machine (shipped pinned to `3.0.55`;
  the event parser was validated against it, so treat a bump as a deliberate change: raise it,
  re-run the tests and a live eval, then commit). `null` installs `@latest`. By default every `cline`
  call Gym makes sets `CLINE_NO_AUTO_UPDATE=1`, so Cline's self-update cannot replace the pinned
  install mid-run; an explicit `env: {CLINE_NO_AUTO_UPDATE: "0"}` re-enables it

See `configs/cline_local_agent.yaml` and `configs/cline_agent.yaml`.

## Trajectory mapping

| Cline `--json` record | Gym item |
|---|---|
| `agent_event` `content_start` / `content_end` (`text`) | assistant message |
| `agent_event` `content_start` (`tool`) | `function_call` |
| `agent_event` `content_end` (`tool`) | `function_call_output` (its `error` when the tool failed) |
| `agent_event` `content_*` (`reasoning`) | `<think>` block on that turn's message |
| `agent_event` `usage`, `run_result` | token usage |
| `hook_event` | ignored (duplicates tool boundaries the agent events already carry) |

Cline closes a turn's reasoning *after* its text, so the think block is assembled from the reasoning
streamed so far rather than from the trailing `content_end`, which would otherwise attach it to the
next turn. `run_result`'s `aggregateUsage` is authoritative when present; a run killed on timeout
never prints it, so on the host path the per-turn `usage` totals are used and `timed_out` is
recorded. Agent sessions report that run's usage as unknown instead (see
[Observations](#observations)).

## Limitations

- Eval-only: no token IDs or logprobs, so this harness is not usable for training as-is.
- One prompt per request. Multi-turn conversation input is reduced to the last user message (plus
  any system message), matching the other CLI agent harnesses.
- A prompt with no whitespace is padded with a trailing space: Cline decides a bare argument was
  quoted by looking for whitespace in it, and rejects a single-word prompt as
  `Unknown command or unquoted prompt` even after `--`.
- Cline 3.0.55 reads a prompt only from argv or a pipe, not from a redirected file, so the prompt is
  an argument: an agent session refuses one over 128 KiB (Linux's limit for one argument), and a
  process listing inside the sandbox shows it.
- The process supervisor's `timeout` bounds the entire run, including model requests and tools.
  Do not rely on a fixed 300-second per-request limit: streamed calls through Gym have remained in
  flight longer than that while Gym sends keepalives and waits for a complete backend response.
  If Cline itself reports `The operation timed out.`, the session reports `model_transport_error`,
  which the Environment Server retries by default. Setting `timeout` in Cline 3.0.55's provider
  settings fails to connect; use the agent's `timeout` for the run's wall-clock budget.
- Agent sessions use only Cline's built-in tools: MCP and `direct_http` grants are not wired into
  Cline, so required grants are refused and optional ones skipped (`resources_tool_transports: []`).
- Cline 3.0.55 always records checkpoint commits under `refs/cline/checkpoints/` when the workdir is a
  git repository, and has no flag to turn them off. The working tree and branches are unchanged, but
  a verifier that reads every ref or unreachable object of the task repository sees them.
- A new `cline_version` needs its platform tarballs' SHA-512 added to `CLINE_RUNTIME_SHA512` before
  agent sessions accept it.
- The host path reports `input_tokens` as Cline's `inputTokens` plus `cacheReadTokens`, which counts
  cache reads twice for the `openai-compatible` provider that the model-server path uses; agent sessions
  do not.
