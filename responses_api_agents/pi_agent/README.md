# Pi Agent

Pi runs the [Pi CLI](https://github.com/earendil-works/pi) inside the task sandbox supplied
by the benchmark's Resources Server. The Environment Server coordinates setup, agent execution,
verification, and cleanup. Pi borrows the sandbox and runs its own model/tool loop there.

[configs/pi_agent.yaml](configs/pi_agent.yaml) is the default agent configuration. Benchmark
and agent settings are independent; the run configuration connects them through the
Environment Server. This harness is currently for evaluation; token IDs and logprobs are not wired up.

## Configure and run

The benchmark owns task data, preparation, verification, and task sandbox settings.
The harness owns its runtime and model/tool loop. The Environment Server binds the two
and closes the agent before verification.

For Terminal-Bench 2.1, see the [single-task run commands](../../benchmarks/terminal_bench_2_1/README.md#pi-in-a-task-sandbox).

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

## Runtime and model requirements

Use one agent worker, an exact `pi_version` (default **0.80.2**), and a Gym `model_server`.
For a Resources-owned task sandbox, supply direct `SandboxAccess` with an absolute
task working directory. Without access, configure agent-owned creation as noted below.
Session setup rejects `pi_version: latest`. Pi's `resources_server` setting is required
only for its compatibility `/run`, not agent sessions.

Supported task images are Linux x86_64/aarch64 glibc or x86_64 musl/Alpine with Python 3.8+,
bash, tar/gzip, and SHA-256 utilities. Session setup uses POSIX sh to install missing Python/Bash
with apt-get or apk when running as root, before launching the Bash runtime installer;
otherwise preinstall them in the image. Setup fails on either a nonzero exit or a provider error.
Older Alpine images
also need patchelf so Pi's Node can use a private, checksum-verified C++ library without
replacing the task's system library. The pinned Node version has no arm64 musl build.
The provider must support sandbox exec and file upload/download.
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

## Requests and settings

Sandbox session input is one text user message, optionally preceded by a system message. Both sandbox
and local execution combine the configured system prompt, request `instructions`, and input
system message in that order. Sampling and chat-template settings belong on the Gym Model Server.
Sandbox sessions reject unsupported request controls before execution.

Set the agent configuration's `max_output_tokens` for a per-model-call output cap, including
reasoning tokens. An adapter-owned Pi extension applies it as `max_tokens`, preserving any
smaller upstream cap. Limits must be positive JavaScript-safe integers. Request-level
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
   removes session files, and disconnects. A failed or missing cleanup receipt blocks close.
7. Environment Server asks Resources to verify and close the task session. The benchmark
   owns its verification procedure and sandbox teardown.

After a successful harness exit in a borrowed sandbox, task processes remain alive
for verification; Resources stops them when it destroys the sandbox. Services must
not depend on the agent's temporary session files. Timeout, cancellation, failed exits,
and agent-owned sandboxes retain full descendant cleanup.

Harness execution uses `nemo_gym.agent_utils.sandbox_session.SandboxSession`, shared with Hermes.
The adapter stages its worker input and collects its own output. The shared session
uploads the supervisor at activation and owns its control files and log access; the common lifecycle
confirms process cleanup, captures artifacts, then releases the provider connection or
owned sandbox. Concurrent closes share that work, failed cleanup remains retryable,
and interrupted activations keep captured observations after session files are removed.
`supervisor_client.py` handles controller-side launch/stop commands; `process_supervisor.py`
runs inside the sandbox. Benchmark Resources session identity/verdict state stays separate.
Missing worker identity or exit diagnostics become `runtime_info_unavailable` or
`worker_exit_code_unavailable` observation gaps. Positive cleanup confirmation remains
required; the adapter validates the captured terminal events without inventing an exit code.

Each sandbox session supports one activation and a matching episode and rollout identity.
Identical request retries join the running activation or replay its result; changed requests
are rejected. A disconnected HTTP caller does not cancel Pi; session close owns cancellation.
Use a single agent-server worker. It never falls back to a host CLI when sandbox setup
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
- `sandbox_bash_timeout_seconds`: maximum runtime of each sandbox bash tool call (default 900);
  shorter tool-requested deadlines are preserved
- `session_close_timeout_seconds`: sandbox process cleanup timeout (default 60)

## Results and limits

Responses retain thinking text, tool calls/results, and Pi-reported usage, including cached
input tokens. Model limits and timeouts preserve gradable partial work after successful agent close.
Provider/API and runtime failures propagate as execution failures and do not enter verification;
they must not be reported as incorrect solutions. Close still returns captured observations.
Pi does not set reward or masking policy.

Response metadata includes `harness_execution: sandbox`, hostname, supervisor PID, and
the pinned Pi version. Model-call references and tool observations are returned by agent close.
Tool timestamps are supervisor receipt times, not executor timestamps; unsupported evidence
is explicitly marked as gaps.

Session and close retries use Gym's shared agent-session implementation. Session state is
process-local. The Environment Server owns normal cleanup and the sandbox provider owns
expiry; Pi does not run a separate session-expiry timer. Failed cleanup remains retryable
and blocks verification. Cleanup is cooperative, not a security boundary against sandbox code.

## Local CLI compatibility

Local settings retain `output_token_policy` (`fixed` or `remaining_context`),
`auto_compaction`, and an optional per-call `bash_timeout`. The dedicated
`pi_sandboxed_agent` also uses this local adapter inside its sandbox and supplies
`mcp_servers` for authenticated Gym tools. Sandbox sessions continue to reject
required Resources tool access rather than silently ignoring it.

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
