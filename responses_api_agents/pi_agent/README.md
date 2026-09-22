# Pi Agent

Runs the [pi](https://github.com/earendil-works/pi) CLI (`pi --print --mode json --no-session`).
pi runs its own tools internally. Resources server for verifier.

Minimal, meant to be modified if needed, and currently eval-only. Token IDs and logprobs are not wired up and
inference can be routed through a Gym model server with `model_server`.

Native sessions run Pi inside the Resources-owned task sandbox; see [Native sandbox sessions](#native-sandbox-sessions).
Calls without an agent session retain the existing local CLI path described below.

## Local CLI quick start

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

## Model id

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

Alternatively, set `model_server` to a Gym model server and set `model` to its served model id. The
agent creates the Pi provider entry automatically. Without `model_server`, the existing provider
configuration is unchanged.

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: the pi command, split on spaces so a multi-word launcher works
- `model`: `<provider>/<model-id>` (see Model id)
- `model_server`: optional Gym model server used to generate the provider entry
- `context_window`: context limit for a generated model entry
- `max_output_tokens`: output limit for a generated model entry
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `workspace_root`: where per-request HOMEs are created and deleted
- `thinking`: passed to `--thinking` (off, minimal, low, medium, high, xhigh)
- `system_prompt`: appended via `--append-system-prompt`
- `timeout`: seconds for the `pi` run
- `extra_args`: extra flags appended to the `pi` command
- `models_config`: written to `~/.pi/agent/models.json`
- `pi_version`: npm version to pin on install (null means latest on the local path; native sessions require an exact version)
- `resources_server`: required only for the agent's existing `/run` endpoint, not for native sessions or direct `/v1/responses`
- `sandbox_install_timeout_seconds`: native runtime installation timeout (default 600)
- `session_close_timeout_seconds`: native process cleanup timeout (default 60)

See `configs/pi_agent.yaml`.

## Native sandbox sessions

The native session path runs the Pi CLI itself inside the task sandbox, with the
Resources-provided working directory (`/app` for SWE-bench Pro). The agent server
remains outside. It installs and launches Pi, collects its JSON events, and confirms
process cleanup; it does not own or stop the sandbox.

This path is experimental. A runnable harness is not an accuracy baseline. Pin the
model, Pi version, context window, thinking level, and output limit before comparing
rewards. Do not assume parity with existing local Pi rollouts.

### Lifecycle and ownership

1. EnvironmentServer asks Resources to seed a task. Resources creates the sandbox.
2. EnvironmentServer passes `SandboxAccess` to Pi's `/v1/agent_sessions` endpoint.
3. Pi connects as a borrower and automatically runs [install_pi_runtime.sh](install_pi_runtime.sh)
   to install Node 22.19.0 and the configured Pi package outside the task repository.
   This installs the harness runtime, not task dependencies or the test environment.
4. EnvironmentServer calls the rollout-prefixed `/v1/responses` route with the agent-session cookie.
5. Pi runs in the sandbox, uses its built-in tools, and sends Chat Completions to
   the rollout-prefixed Gym model-server URL. The sandbox must be able to reach that URL.
6. Agent close confirms supervisor and descendant cleanup, returns observations,
   removes session files, and disconnects. A failed or missing cleanup receipt blocks close.
7. EnvironmentServer asks Resources to verify and close the task session. For SWE Pro,
   Resources extracts the patch and grades it in a fresh verification sandbox.

The native session supports one activation, a matching episode and rollout identity,
and a single agent-server worker. It never falls back to a host CLI when sandbox setup
fails. Calls without an agent session retain the local Pi behavior; they do not operate
on the Resources-owned task sandbox.

Server startup does not install Pi on the host. Host installation happens only on the
first local invocation. Native sessions install their runtime during session initialization,
using the same seed/close contracts as Hermes. The shell script is an internal implementation
detail, not a separate endpoint or a setup step users must run.

### Configure and run

Use [pi_swe_pro_native.yaml](configs/pi_swe_pro_native.yaml) as a composition example.
Combine it with the SWE Pro resources configuration, a sandbox provider, and a Gym
model-server instance named `policy`; set `pi_swe_pro_agent.model` to the served model ID
using Gym's normal server configuration nesting. Only EnvironmentServer needs the Resources
reference in this composition; Pi's `resources_server` setting is needed only for its existing `/run`.

The sample uses Pi `0.80.2`. Native seed requires an exact `pi_version`, not `latest`.
Supported task images are Linux x86_64/aarch64 glibc with Python 3.9+, bash, curl, tar/xz,
and SHA-256 utilities. The provider must implement PTY process sessions, including exit
acknowledgement and signalling. Installation needs network access to nodejs.org and npm.
Musl/Alpine images are not supported.

The Node runtime, Pi package, and isolated HOME are outside the task repository. Pi's
Node is addressed by absolute path; it does not replace the task's Python or Node on PATH.
Project extensions, skills, prompt templates, and themes are disabled; repository context
files may still be read by Pi.

Keep `resources_tool_transports: []` for SWE Pro: Pi provides its own sandbox tools.
Required Resources HTTP/MCP tools are rejected. Native sessions also reject host command,
extra-argument, and environment overrides; those remain available on the local path.

After starting the composed servers, submit a materialized native task to EnvironmentServer's
`/run` endpoint, not the agent's `/run`. For example, with `swe-task.json` containing one row
produced by [the SWE Pro preparation script](../../benchmarks/swebench/pro/prepare.py),
including its prepared prompt, verification scripts, and task fields:

```bash
jq '{
  episode_id: {rollout_id: "pi-smoke", attempt: 1},
  task: {
    task_id: {taskset: "swebench-pro", task_id: .instance_id},
    task_input: {
      responses_create_params: .responses_create_params,
      task_data: del(.responses_create_params, .agent_ref)
    }
  }
}' swe-task.json > pi-episode.json

curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @pi-episode.json "${PI_ENVIRONMENT_URL}/run"
```

Input is one text user message, optionally preceded by a system message. The adapter also
applies `instructions` and its configured system prompt. It honors `max_output_tokens`
through Pi's model configuration. Unsupported sampling, history, and tool-policy overrides
are rejected rather than silently ignored; configure sampling and chat-template settings
on the Gym model server.

### Results and limitations

Responses retain thinking text, tool calls/results, and Pi-reported usage, including cached
input tokens. A failed or timed-out invocation preserves partial output. Successful agent
close is required before the benchmark decides whether partial work earns reward.
Pi does not set reward or masking policy.

Response metadata includes `harness_execution: sandbox`, hostname, supervisor PID, and
the pinned Pi version. Model-call references and tool observations are returned by agent close.
Tool timestamps are supervisor receipt times, not executor timestamps; unsupported evidence
is explicitly marked as gaps.

Cleanup is cooperative, not a security boundary against hostile sandbox code. The worker
retains the last 64 close receipts for immediate retries, not durable recovery. Agent sessions
are process-local and do not survive server crashes. Resources and the sandbox provider
retain responsibility for owner-side cleanup and expiry.
