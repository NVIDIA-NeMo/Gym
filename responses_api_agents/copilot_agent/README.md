# Copilot Agent

Copilot runs the [GitHub Copilot CLI](https://github.com/github/copilot-cli) inside the task sandbox
supplied by the benchmark's Resources Server. The Environment Server coordinates setup, agent execution,
verification, and cleanup. Copilot borrows the sandbox and runs its own model/tool loop there.

Copilot reaches the model through its bring-your-own-key (BYOK) provider support, pointed at the
rollout-prefixed Gym Model Server with `COPILOT_OFFLINE=true`. It needs no GitHub account or token and makes
no Copilot service calls. Authentication, client telemetry, built-in MCP servers, and auto-update are disabled; task shell tools can still access the network. Built-in web fetch is denied by default; set `allow_web_fetch: true` to enable it.

[configs/copilot_agent.yaml](configs/copilot_agent.yaml) is the default agent configuration.
Capability level: Observability, with task sandboxes, within the limits described below. Resources tool grants
and training-token capture are not supported. Subagent hierarchy and compaction observations are reported as gaps.

## Configure and run

The adapter runs only in agent sessions, so episodes go through an Environment Server. For Terminal-Bench
2.1, save this composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/copilot_agent/configs/copilot_agent.yaml
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
        name: copilot_agent
      resources_tool_transports: []
```

Supply the `policy_model` Gym Model Server, `policy_model_name`, and the `sandbox` provider as for the
other harnesses in [the Terminal-Bench 2.1 README](../../benchmarks/terminal_bench_2_1/README.md). The
sandbox must be able to reach the Model Server. Terminal-Bench 2.1 and SWE-bench Pro qualification is
still pending, so these pairings require `--allow-unsupported-pairing` during development. Copilot is
not yet declared in their `allowed_agents`.

For SWE-bench Pro, replace the Resources config import with
`resources_servers/swebench_pro/configs/swebench_pro.yaml`, bind the `swebench_pro` taskset to
`single_agent_turn`, and set `resources_server.name: swebench_pro_resources_server`. Keep the same
Copilot Agent Server and `resources_tool_transports: []`. Prepare its data using
[the SWE-bench Pro instructions](../../resources_servers/swebench_pro/README.md), and declare
`taskset: swebench_pro` on that dataset so Gym materializes typed tasks for the route.

## Runtime and model requirements

Use an exact `copilot_version` (default **1.0.103**). Session setup reads the sandbox's OS, CPU and C library
with `uname` and `getconf`, and the Agent Server selects the npm platform package
`@github/copilot-linux-{x64,arm64}` (glibc 2.28 or newer, with libstdc++) or
`@github/copilot-linuxmusl-{x64,arm64}`. A sandbox that is not Linux, has another CPU, or has an older glibc is
refused at seed. The Agent Server reads the package's tarball URL and sha512 integrity from `npm_registry`
(default `https://registry.npmjs.org`). [install_copilot_runtime.sh](install_copilot_runtime.sh) then downloads
the tarball in the sandbox into `/tmp/nemo-gym-copilot-<version>`, outside the task repository, checks it against
that digest, unpacks the standalone binary, and runs it once with a private HOME so that its bundled package lands
in the runtime directory instead of the task's HOME. A later seed in the same sandbox reuses a runtime that has its
ready marker and reports the pinned version. Concurrent seeds in one sandbox share one installation when the image
has `flock`.

The default download path requires registry access from both the Agent Server and the sandbox, plus a CA bundle.
For a sandbox without registry egress (including SWE-bench Pro with `task_network_policy`), preinstall the
pinned runtime in the image and set all three fields: `copilot_binary` (absolute executable path),
`copilot_binary_sha256` (lowercase SHA-256 of that binary), and `copilot_dist_dir` (absolute path to its unpacked
version-matched package). Seed verifies the binary digest before executing it, verifies the reported version,
and uses the supplied package directly without querying npm or invoking the installer. The image and unpacked
package remain trusted operator inputs; the binary checksum does not authenticate the package directory.
This path needs preinstalled bash and `sha256sum`. The default download path needs bash, `curl` or `wget`, `sha512sum`,
and `tar` with gzip; it does not need Python. Session setup installs bash, and curl with ca-certificates when
neither downloader is present, with apt-get or apk when running as root; otherwise preinstall them. The provider
must support sandbox exec and file upload/download.

Copilot's own state, logs, and session store live in the session directory (`COPILOT_HOME`); the bundled
package is read through `COPILOT_CLI_DIST_DIR`. Tool shells keep the task's HOME and XDG directories.
Copilot 1.0.103 still writes a small device-ID file under the task's HOME on each run.

The model must support tool calling. Copilot sends non-streaming Chat Completions (`--stream off`) with
its own sampling: `temperature: 0`, `top_p: 0.95`, and zero frequency and presence penalties (observed with
1.0.103). Pin the sampling you want with the vLLM Model Server's `sampling_overrides`.

## Requests and settings

An activation accepts one text user message. Copilot has no system-prompt option, so `instructions`
and system messages are rejected, as are request-level sampling controls and the total
`max_output_tokens` budget. The prompt is passed as one command-line argument, `--prompt=<text>`, so a prompt
with a NUL character or of 128 KiB or more is rejected. `metadata.agent_timeout_sec` is applied, capped by
`timeout`.

- `context_window`: the model's context length, which holds the prompt and the output. Copilot's prompt-token
  limit, `COPILOT_PROVIDER_MAX_PROMPT_TOKENS`, is `context_window` minus `max_output_tokens`, which must be
  smaller. Unset uses Copilot's built-in default for an unrecognized model.
- `max_output_tokens`: per-model-call output limit, sent as `max_tokens`. Unset sends none.
- `reasoning_effort`: passed to `--reasoning-effort`.

Copilot runs with `--allow-all-tools --allow-all-paths --no-ask-user`; the task sandbox is the isolation
boundary. It keeps its built-in tools (`bash`, `view`, `create`, `edit`, `grep`, `glob`, subagents through
`task`, and others). Resources Server tool grants are not used: a required grant is refused at seed, so
keep `resources_tool_transports: []`. Custom instructions in the task repository, such as `AGENTS.md`,
are loaded as Copilot normally does.

## Lifecycle and ownership

1. The Environment Server asks Resources to seed a task. Resources creates the sandbox.
2. The Environment Server passes `SandboxAccess` to `/v1/agent_sessions`.
3. Copilot connects as a borrower and installs or reuses its runtime.
4. The Environment Server calls the rollout-prefixed `/v1/responses` with the agent-session cookie.
5. The shared `harness_runner.sh` runs `copilot --prompt=<text>` under the shared process supervisor,
   `process_supervisor.sh`, and records each JSON event line with the time it was seen. Both run with bash.
   Every model call goes to the rollout-prefixed Gym Model Server.
6. Agent close confirms harness termination, returns observations, removes session files, and disconnects.
7. The Environment Server asks Resources to verify and close the task session.

Harness execution uses `nemo_gym.agents.utils.sandbox_session.SandboxSession`. Each session supports one
activation; identical retries join it and changed requests are rejected. A disconnected caller does not
cancel Copilot; session close owns cancellation. Without `SandboxAccess`, configure `sandbox_provider`
and `sandbox_config` for an agent-owned sandbox (default workdir `/app`). Session setup, including the start of an owned sandbox, must finish within `sandbox_install_timeout_seconds`. Owned sandboxes default to a TTL of the install budget plus the run budget plus 600 seconds; an explicit `ttl_s`, including `null`, is preserved. Check that your provider enforces TTLs. Calls without an agent session and
the compatibility `/run` are rejected: Copilot never runs on the Agent Server host.

## Results and limits

Copilot masks its BYOK provider key and header values as `******` wherever they appear in the events it writes,
including model text, tool output, tool-call IDs and timestamps. The adapter therefore sends a random unused key
and a random per-session `x-session-id`, which is also the invocation ID in close observations and model-call
captures; the rollout ID stays in the Model Server URL. Task text cannot contain either value, so the recorded
output matches what the model saw, except that these two values stay masked if a tool prints Copilot's environment.

Responses keep the root agent's reasoning, messages, tool calls, and tool results. Child-agent events
do not enter the root conversation; their model-call references and usage remain included when available.
Usage comes from Copilot's
OpenTelemetry chat spans, written locally to the session directory: counts are summed over successful
model calls, and a count the provider omitted stays unknown rather than zero. Close returns the model-call
references, harness tool timing, sandbox identity and execution outcome, and explicit observation gaps.
Observed rejected model calls contribute zero completed-response tokens; missing chat telemetry stays unknown.
A call that Copilot's events report but that has no chat span also makes aggregate usage unknown
(`model_usage_spans_missing`); its model-call reference comes from the events.
Copilot 1.0.103 omits compaction calls from its OpenTelemetry chat spans. The adapter also checks its persisted
session events, including late background compaction attempts with no subsequent chat span. Aggregate
response usage stays unknown after observed compaction (`compaction_model_usage_unavailable`) or when
Copilot's session event store cannot be inspected (`compaction_state_unavailable`). Enable Gym Model
Server capture to retain per-call token counts, including compaction. An unreadable telemetry line, such as a
span cut off mid-write, also makes aggregate usage unknown (`model_usage_telemetry_malformed`).

A provider error that Copilot reports (`session.error`) returns a failed response with a classified cause:
an HTTP status becomes `model_request_error`, and an unreachable provider becomes `model_transport_error`.
As in Copilot's own exit status, errors from sub-agents or hooks, and a model-call error followed by a
later root assistant message, do not fail the run.
Root session errors, including recovered model-call errors, remain in the canonical failure report
forwarded by activation and close. These observations do not select an episode failure or request another attempt.
Child conversations and tool observations are not projected into the root invocation; the
`subagent_hierarchy_unavailable` gap records this limitation. Child model-call references and usage remain retained.
The Environment Server decides grading and retries. Copilot retries 429 and 5xx responses itself before
reporting. A timeout returns partial work as incomplete. Other runner or termination failures are execution
errors. A deadline can interrupt a tool before Copilot reports its result. That tool remains incomplete,
with its observed start and a timing gap; the adapter does not invent a terminal output. Such a partial
rollout can be graded. The retained-artifact checker accepts the incomplete tool and its absent output;
other unavailable evidence, such as complete tool timing, remains an explicit gap.

Response metadata includes `harness_execution: sandbox`, hostname, runner PID, and `copilot_version`.

Owned sandboxes support `network_access: inherit` (default), `model_only`, or `model_and_tools`.
Restricted values require the preinstalled runtime and OpenSandbox, and allow only the Model Server host
(Copilot keeps no external tool grants). Policies open every port and path on that host, including co-hosted
services. Restricted policies on borrowed sandboxes, conflicting explicit provider policies, unsupported
providers, and unreachable model endpoints are refused before sandbox creation. Borrowed sandboxes retain
the Resources Server's policy. This setting limits egress; denying built-in web fetch alone does not restrict
shell networking. Restricted OpenSandbox enforcement has not been exercised by the unit tests.
