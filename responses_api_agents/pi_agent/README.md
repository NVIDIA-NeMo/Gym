# Pi Agent

Pi runs the [Pi CLI](https://github.com/earendil-works/pi) inside the task sandbox supplied
by the benchmark's Resources Server. The EnvironmentServer coordinates setup, agent execution,
verification, and cleanup. Pi borrows the sandbox and runs its own model/tool loop there.

[configs/pi_agent.yaml](configs/pi_agent.yaml) is the default agent configuration. Benchmark
and agent settings are independent; the run configuration connects them through the
EnvironmentServer. This harness is currently for evaluation; token IDs and logprobs are not wired up.

## Configure and run

Import [configs/pi_agent.yaml](configs/pi_agent.yaml) in your environment/run configuration
and compose its `pi_agent` with your benchmark's Resources server, a sandbox provider, and a Gym
model server. The agent config contains no benchmark or dataset selection. Use the existing
component names directly; the run configuration supplies the EnvironmentServer references.

- On Pi, set `num_workers: 1`, an exact `pi_version` (for example `0.80.2`),
  `model_server` pointing to the Gym model server, and `model` to its served model ID.
- On [single-agent EnvironmentServer](../../environment_servers/single_agent_turn/configs/single_agent_turn.yaml),
  set `agent_server` to Pi, `resources_server` to the benchmark, and `resources_tool_transports: []`.
- Resources must support sandbox sessions and return direct `SandboxAccess` with an
  absolute task working directory. Pi does not create a fallback sandbox.

Benchmark selection and evaluation settings belong in that environment/run configuration,
not a Pi-specific benchmark preset. Pi's own `resources_server` setting is needed only for
its existing `/run`; omit it for sandbox sessions. Session setup rejects `pi_version: latest`.

Supported task images are Linux x86_64/aarch64 glibc or x86_64 musl/Alpine with Python 3.8+,
bash, tar/gzip, and SHA-256 utilities. Missing bootstrap packages are installed with apt-get
or apk when running as root; otherwise preinstall them in the image. Older Alpine images
also need patchelf so Pi's Node can use a private, checksum-verified C++ library without
replacing the task's system library. The pinned Node version has no arm64 musl build.
The provider must implement PTY process sessions, including exit acknowledgement and signalling.
Installation needs network access to nodejs.org and npm; musl also uses unofficial-builds.nodejs.org
and, for the older C++ runtime fallback, dl-cdn.alpinelinux.org.

The Node runtime, Pi package, and isolated HOME are outside the task repository. Pi's
Node is addressed by absolute path; it does not replace the task's Python or Node on PATH.
Sandbox sessions bound each bash command and advise the agent to search repository/local
dependency directories rather than shared mounts. This returns control after a stalled tool;
the separate `timeout` still bounds the whole agent episode. Installer errors include both
stdout and stderr so provider status messages do not hide the actual failure.
Project extensions, skills, prompt templates, and themes are disabled; repository context
files may still be read by Pi.

Keep `resources_tool_transports: []`: Pi provides its own sandbox tools.
Required Resources HTTP/MCP tools are rejected. Sandbox sessions also reject host command,
extra-argument, and environment overrides; those remain available on the local path.

For example, a SWE-bench Pro run can use the following `run.yaml`. The benchmark's
[Resources config](../../resources_servers/swebench_pro/configs/swebench_pro.yaml)
owns sandbox and verification settings; Pi's config owns the harness settings.
The run binds these independent components:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/pi_agent/configs/pi_agent.yaml
  - environment_servers/single_agent_turn/configs/single_agent_turn.yaml

environment_routing_mode: taskset
environment_server_routes:
  swebench_pro: single_agent_turn

single_agent_turn:
  environment_servers:
    single_agent_turn:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: pi_agent
```

Supply a `policy_model` model server and `sandbox` provider in `model-provider.yaml`,
and set `pi_agent.responses_api_agents.pi_agent.model` to the served model ID there.
Prepare typed input from the existing benchmark rows before collection. Use the same
run configuration to launch servers and collect rollouts:

```bash
python benchmarks/swebench/pro/materialize_single_agent_tasks.py prepared.jsonl tasks.jsonl --taskset swebench_pro
gym env start --config run.yaml --config model-provider.yaml
gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  -i tasks.jsonl -o rollouts.jsonl
```

The collector calls the EnvironmentServer's `/run`; the environment seeds Resources
and Pi, invokes Pi's `/v1/responses`, closes Pi, then verifies and closes Resources.
Collection does not call Pi's compatibility `/run` endpoint.

After starting the composed servers, submit an episode request to EnvironmentServer's
`/run` endpoint, not the agent's `/run`. Use the benchmark's prepared task: `task.task_input`
contains `responses_create_params` for Pi and `task_data` matching that Resources server's
seed contract. The request also contains `episode_id` and `task.task_id` for rollout identity.
For a request saved as `episode.json`:

```bash
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @episode.json "${PI_ENVIRONMENT_URL}/run"
```

Input is one text user message, optionally preceded by a system message. The adapter also
applies `instructions` and its configured system prompt. For sandbox sessions, request
`max_output_tokens` overrides the agent's configured default. An adapter-owned Pi extension
sends the limit as `max_tokens` on every Chat Completions request and preserves any smaller
upstream limit. This is a per-model-call cap, including reasoning tokens, rather than a total
episode budget. Limits must be positive JavaScript-safe integers. Model-server configuration
must not override this request limit with a larger value. Enforcement is tested with Pi 0.80.2.
Unsupported sampling, history, and tool-policy overrides are rejected rather than silently
ignored; configure sampling and chat-template settings on the Gym model server.

## Lifecycle and ownership

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
7. EnvironmentServer asks Resources to verify and close the task session. The benchmark
   owns its verification procedure and sandbox teardown.

Each sandbox session supports one activation, a matching episode and rollout identity,
and a single agent-server worker. It never falls back to a host CLI when sandbox setup
fails. Calls without an agent session retain the local Pi behavior; they do not operate
on the Resources-owned task sandbox.

Server startup does not install Pi on the host. Host installation happens only on the
first local invocation. Sandbox sessions install their runtime during session initialization,
using the same seed/close contracts as Hermes. The shell script is an internal implementation
detail, not a separate endpoint or a setup step users must run.

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: local compatibility launcher; task sessions use their installed Pi runtime
- `model`: the served model ID for the configured Gym model server
- `model_server`: Gym model server used to generate the Pi provider entry
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
- `sandbox_bash_timeout_seconds`: maximum runtime of each sandbox bash tool call (default 900);
  shorter tool-requested deadlines are preserved
- `session_close_timeout_seconds`: sandbox process cleanup timeout (default 60)

## Results and limits

Responses retain thinking text, tool calls/results, and Pi-reported usage, including cached
input tokens. A failed or timed-out invocation preserves partial output. Successful agent
close is required before the benchmark decides whether partial work earns reward.
Pi does not set reward or masking policy.

Response metadata includes `harness_execution: sandbox`, hostname, supervisor PID, and
the pinned Pi version. Model-call references and tool observations are returned by agent close.
Tool timestamps are supervisor receipt times, not executor timestamps; unsupported evidence
is explicitly marked as gaps.

Cleanup is cooperative, not a security boundary against hostile sandbox code. Like Hermes,
successful close receipts are retained for `session_close_retry_window_seconds` (default 300),
starting after cleanup. Set it to cover the caller's retry horizon. Other sessions and retries
do not shorten or extend that window. Expired receipts are pruned on seed/close activity;
stale activations still cannot fall back to the host. Sessions and receipts are process-local,
not durable recovery. Resources and the sandbox provider own sandbox cleanup and expiry.

Session IDs come from EnvironmentServer. Repeating an identical seed returns the
same session, while reusing its ID for a different seed fails. Close accepts an explicit
session ID and episode without a cookie, so a lost seed response can still be cleaned up.
Close-before-seed records a tombstone to reject delayed creation. The agent uses a separate
random filesystem directory, so caller IDs never become paths. `session_lifetime_seconds`
(default 21600) bounds abandoned sessions through the same confirmed-cleanup path. Failed
cleanup retains the closing state for retry and blocks another activation.

## Local CLI compatibility

Calls without an agent session retain local CLI compatibility. They do not use the
Resources-owned task sandbox. Configure a Gym `model_server`, or supply an explicit Pi provider
configuration as below. The self-contained math example keeps its local execution settings.

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

For Gym-managed inference, set `model_server` to a Gym model server and set `model` to its served model id. The
agent creates the Pi provider entry automatically. Without `model_server`, the existing provider
configuration is unchanged.
