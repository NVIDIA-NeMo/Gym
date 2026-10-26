# Copilot Agent

Copilot runs the [GitHub Copilot CLI](https://github.com/github/copilot-cli) inside the task sandbox
supplied by the benchmark's Resources Server. The Environment Server coordinates setup, agent execution,
verification, and cleanup. Copilot borrows the sandbox and runs its own model/tool loop there.

Copilot reaches the model through its bring-your-own-key (BYOK) provider support, pointed at the
rollout-prefixed Gym Model Server with `COPILOT_OFFLINE=true`. It needs no GitHub account or token and makes
no Copilot service calls. Authentication, client telemetry, built-in MCP servers, and auto-update are disabled; task tools can still access the network.

[configs/copilot_agent.yaml](configs/copilot_agent.yaml) is the default agent configuration. This harness
implements evaluation and observability with task sandboxes. Resources tool grants and training-token capture are not supported. Subagent hierarchy and compaction observations are reported as gaps.

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
sandbox must be able to reach the Model Server. Terminal-Bench 2.1 and SWE-bench Pro declare
Copilot in `allowed_agents`; other pairings require `--allow-unsupported-pairing` during development
and real-model qualification before declaration.

For SWE-bench Pro, replace the Resources config import with
`resources_servers/swebench_pro/configs/swebench_pro.yaml`, bind the `swebench_pro` taskset to
`single_agent_turn`, and set `resources_server.name: swebench_pro_resources_server`. Keep the same
Copilot Agent Server and `resources_tool_transports: []`. Prepare its data using
[the SWE-bench Pro instructions](../../resources_servers/swebench_pro/README.md), and declare
`taskset: swebench_pro` on that dataset so Gym materializes typed tasks for the route.

## Runtime and model requirements

Use an exact `copilot_version` (default **1.0.103**). Session setup installs the npm platform package
`@github/copilot-linux-{x64,arm64}` (glibc 2.28 or newer, with libstdc++) or
`@github/copilot-linuxmusl-{x64,arm64}` into `/tmp/nemo-gym-copilot-<version>`, outside the task repository.
[install_copilot_runtime.py](install_copilot_runtime.py) verifies the tarball against the registry's sha512
integrity, unpacks the standalone binary, and runs it once with a private HOME so that its bundled package
lands in the runtime directory instead of the task's HOME. A later seed in the same sandbox reuses a runtime
that reports the pinned version. Installation needs network access to `npm_registry`
(default `https://registry.npmjs.org`) and a CA bundle in the image.

The sandbox needs `python3` for the process supervisor and the installer. Session setup installs it with
apt-get or apk when running as root; otherwise preinstall it. The provider must support sandbox exec and
file upload/download.

Copilot's own state, logs, and session store live in the session directory (`COPILOT_HOME`); the bundled
package is read through `COPILOT_CLI_DIST_DIR`. Tool shells keep the task's HOME and XDG directories.
Copilot 1.0.103 still writes a small device-ID file under the task's HOME on each run.

The model must support tool calling. Copilot sends non-streaming Chat Completions (`--stream off`) with
its own sampling: `temperature: 0`, `top_p: 0.95`, and zero frequency and presence penalties (observed with
1.0.103). Pin the sampling you want with the Model Server's `sampling_overrides`.

## Requests and settings

An activation accepts one text user message. Copilot has no system-prompt option, so `instructions`
and system messages are rejected, as are request-level sampling controls and the total
`max_output_tokens` budget. `metadata.agent_timeout_sec` is applied, capped by `timeout`.

- `context_window`: prompt-token limit, passed as `COPILOT_PROVIDER_MAX_PROMPT_TOKENS`. Unset uses
  Copilot's built-in default for an unrecognized model.
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
5. `sandbox_runner.py` runs `copilot -p` under the shared process supervisor and records its JSONL events.
   Every model call goes to the rollout-prefixed Gym Model Server.
6. Agent close confirms harness termination, returns observations, removes session files, and disconnects.
7. The Environment Server asks Resources to verify and close the task session.

Harness execution uses `nemo_gym.agent_utils.sandbox_session.SandboxSession`. Each session supports one
activation; identical retries join it and changed requests are rejected. A disconnected caller does not
cancel Copilot; session close owns cancellation. Without `SandboxAccess`, configure `sandbox_provider`
and `sandbox_config` for an agent-owned sandbox (default workdir `/app`). Owned sandboxes default to a TTL of the install budget plus the run budget plus 600 seconds; an explicit `ttl_s`, including `null`, is preserved. Check that your provider enforces TTLs. Calls without an agent session and
the compatibility `/run` are rejected: Copilot never runs on the Agent Server host.

## Results and limits

Responses keep Copilot's reasoning, messages, tool calls, and tool results. Usage comes from Copilot's
OpenTelemetry chat spans, written locally to the session directory: counts are summed over successful
model calls, and a count the provider omitted stays unknown rather than zero. Close returns the model-call
references, harness tool timing, sandbox identity and execution outcome, and explicit observation gaps.
Observed rejected model calls contribute zero completed-response tokens; missing chat telemetry stays unknown.
Copilot 1.0.103 omits compaction calls from its native usage spans. After observed compaction, aggregate
response usage is unknown and `compaction_model_usage_unavailable` explains the gap. Enable Gym
Model Server capture to retain the complete per-call token counts, including compaction.

A provider error that Copilot reports (`session.error`) returns a failed response with a classified cause:
an HTTP status becomes `model_request_error`, and an unreachable provider becomes `model_transport_error`.
The Environment Server decides grading and retries. Copilot retries 429 and 5xx responses itself before
reporting. A timeout returns partial work as incomplete. Other runner or termination failures are execution
errors.

Response metadata includes `harness_execution: sandbox`, hostname, runner PID, and `copilot_version`.
