# Codex Agent

Runs the OpenAI Codex CLI (`codex exec`) as a NeMo Gym agent server.

## Configure and run

[configs/codex_agent.yaml](configs/codex_agent.yaml) is the default harness definition.
The benchmark owns task data, preparation, verification, and task sandbox settings.
The harness owns its runtime and model/tool loop. The Environment Server binds the two
and closes the agent before verification. Codex borrows the task sandbox and runs its CLI
and shell/file tools inside it.

Run from the Gym repository root with Gym and the benchmark's preparation dependencies
installed. For SWE-bench Pro, save this composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/codex_agent/configs/codex_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: codex_agent
      resources_tool_transports: []
```

Supply the `policy_model` Gym Model Server, `policy_model_name`, and `sandbox` provider in
`model-provider.yaml`. The agent's `model` defaults to `${policy_model_name}` and remains
overridable. The sandbox must be able to reach the Model Server. Keep model sampling and
per-call output limits on the Model Server.

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent codex_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

The collector calls Environment Server `/run`: seed Resources, seed the agent, call its
rollout-prefixed `/v1/responses`, close the agent, verify, then close Resources. Prepared
flat rows use `single_agent_turn_legacy` with this native session lifecycle; no additional
materialization script is needed. Collection does not call the agent's compatibility `/run`.
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

Use one agent worker, a pinned `codex_version` (default **0.144.4**), direct `SandboxAccess`,
and an absolute task workdir disjoint from the adapter directories. Required Resources HTTP/MCP
tools are rejected; Codex supplies its own tools. The task sandbox supplies isolation, so
`sandbox_mode` must be `danger-full-access`. No Codex CLI is required on the agent-server host.

The installer accepts Linux x86_64/aarch64 glibc and x86_64 musl/Alpine with Python 3.8+.
POSIX sh installs missing Python and Bash using apt-get or apk when running as root,
before invoking the Bash runtime installer; otherwise the image must provide them.
The pinned Node 22.19.0 build has no arm64 musl binary.
Older Alpine images use a checksum-verified private C++ library for Node. Runtime files live
under `/tmp/nemo-gym-codex-node-*`, and session HOME/cache/config live under
`/tmp/nemo-gym-codex-sessions/*`. Task Python, Node, libraries, and PATH are preserved.
Installation errors retain the command, exit status, stdout, and stderr. Runtime compatibility
must still be checked on actual task images; a mocked installer test is not image validation.

Codex 0.144.4 uses the streaming **Responses API**. Use a sandbox-reachable Gym Model Server;
Gym adapts Responses to Chat Completions where supported. Requests retain the rollout capture
key, reasoning, and tool history. Codex sends `prompt_cache_key`, which some hosted NIM Chat
endpoints reject. A deployment that removes that option is a separate compatibility bridge,
not evidence of direct endpoint compatibility.

For custom models, set `model_context_window` to the served limit and optionally
`model_auto_compact_token_limit` below it. Both must be positive integers; an explicit
compaction threshold must not exceed 90% of the explicit window. The pinned CLI also clamps
unknown-model metadata to a 272,000-token context and 90% compaction threshold. These are
context-management controls, not generation budgets.

## Requests, results, and lifecycle

Input is a string or one text user message, optionally preceded by a system/developer message.
Configured `system_prompt`, request `instructions`, and that preceding message become Codex
developer instructions in that order. Unsupported conversation/multimodal input, required
external tools, provider overrides, and extra host config are rejected before activation.

Request `max_output_tokens`, `temperature`, `top_p`, reasoning controls, and unsupported tool
policies are rejected rather than ignored. Codex cannot enforce these request-level controls
for a custom Gym provider. Set per-call generation settings on the Model Server and inspect
captured requests to verify the effective values. A configured `reasoning_effort` is likewise
rejected in sandbox sessions. An explicit request model must match the configured model.

Session identity, immutable seed binding, seed/close locking, and close receipts use Gym's
shared agent-session implementation. Setup returns a session cookie only after installation
succeeds. Failed setup with failed cleanup retains cleanup-only state through
`AgentSessionSetupError`: activation/reseeding are rejected, and close can retry.

Each session supports one logical activation. Identical requests join the running invocation
or replay its result/error; changed requests are rejected. A disconnected HTTP waiter does
not cancel the invocation. Session close owns cancellation. Shared successful close receipts
last `session_close_retry_window_seconds` (default 300 seconds) without renewal on retries.
Stale cookies never fall back to host execution. State is process-local; Resources/provider
own sandbox expiry and crash recovery. There is no separate per-agent session-expiry timer.

One shared Linux supervisor per activation fences delayed launches and kills/reaps detached
tool descendants. Close confirms its cleanup receipt before cancelling provider execution,
then removes adapter-owned files and disconnects. Missing/negative cleanup evidence blocks
verification and retains state for retry. The borrowing agent never destroys the sandbox.
Cleanup coordinates lifecycle; it is not a security boundary against hostile task code.

Responses and close observations preserve text, reasoning, tool events, partial output,
CLI aggregate usage, and sandbox hostname/PID/version. Wall-time and the pinned CLI's explicit
model-output-limit event yield incomplete, potentially gradable work. Provider/API/runtime
failures raise execution errors and bypass verification. Only Resources determines reward.

CLI events omit exact model-call join IDs and can omit failed-call usage even after a recovered
retry. Missing/default-zero cache and reasoning details remain unknown, with observation gaps.
Compare CLI totals with captured model calls; do not interpret unavailable usage as zero.
Known model-metadata/compaction warnings remain visible as gaps after clean completion or a gradable model/wall-time limit.

`sandbox_install_timeout_seconds` defaults to 600 and `session_close_timeout_seconds` to 60;
both must be finite and positive. Measure setup time, supervisor process/memory cost, and close
latency before scaling. Inference support does not establish training token-ID/logprob support.

## Legacy direct-agent quick start

### env.yaml

For the OpenAI API:

```yaml
openai_api_key: sk-...
```

For any endpoint that serves the OpenAI Responses API over SSE:

```yaml
openai_api_key: EMPTY
```

and set `openai_base_url` (must include `/v1`; Codex appends `/responses` itself).

### Launch

For a quick eval against OpenAI (or any Responses endpoint set via `openai_base_url`), pass the resources server config, which includes the agent server config:

```bash
gym env start --resources-server reasoning_gym/reasoning_gym_codex_agent
```

#### Against a Gym model server

Every Gym model server serves the streaming Responses dialect Codex speaks on `POST /v1/responses` (`SimpleResponsesAPIModel` sanitizes the request and synthesizes the SSE stream), so Codex can run against any backend Gym serves — vLLM, OpenAI, an inference provider. Set the agent's `model_server` ref to that server (it takes precedence over `openai_base_url`); the harness resolves the provider `base_url` to it.

`reasoning_gym_codex_agent_model_server.yaml` wires the agent's `model_server` ref to `policy_model`. Compose it with any model server (here a vLLM serving `policy_model`):

```bash
gym env start \
  --resources-server reasoning_gym/reasoning_gym_codex_agent_model_server \
  --model-type vllm_model
```

This path needs only the model server's `policy_base_url`, `policy_api_key`, and `policy_model_name` (in `env.yaml` or as `+` overrides) — no `openai_*` vars.

### Run the agent

```bash
gym eval run --no-serve \
    --agent reasoning_gym_codex_agent \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output codex_rollout.jsonl \
    --limit 1
```

For the model-server config above, use `--agent reasoning_gym_codex_agent_model_server`.

### Smoke test

Check the streaming `/v1/responses` dialect and the real-CLI seam without a full rollout. Launch a model server, then take its URL from the `gym env start` log (`'url': 'http://127.0.0.1:<port>'`):

```bash
gym env start --model-type vllm_model \
  +policy_base_url=https://integrate.api.nvidia.com/v1 \
  '+policy_api_key=${oc.env:NVIDIA_API_KEY}' +policy_model_name=meta/llama-3.1-8b-instruct

# 1. the endpoint speaks the streaming Responses dialect:
curl -N $URL/v1/responses -H 'content-type: application/json' \
  -d '{"model":"x","stream":true,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"2+2?"}]}]}'

# 2. the real Codex CLI runs against it:
mkdir -p /tmp/codex_home && cat > /tmp/codex_home/config.toml <<EOF
model_provider = "gym"
[model_providers.gym]
name = "gym"
base_url = "$URL/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
EOF
CODEX_HOME=/tmp/codex_home OPENAI_API_KEY=local \
  codex exec --json --ephemeral --skip-git-repo-check "What is 2+2?" < /dev/null
```

## Description

The agent runs `codex exec --json` as an async subprocess for each request. Codex handles all tool execution (shell commands, file edits, MCP tool calls) internally in a per-rollout scratch working directory. The agent parses the JSONL event stream into NeMoGym output items and forwards the response to a resources server for verification.

Codex talks to the model via the OpenAI Responses API over SSE (`wire_api = "chat"` was removed from Codex). This means it can connect to OpenAI directly, to any endpoint implementing the streaming Responses API, or — via the agent's `model_server` ref — to any NeMo Gym model server, since every Gym model server serves the streaming Responses dialect by sanitizing the request (extra bookkeeping fields, `namespace` tool specs are flattened to plain functions) and re-emitting its complete response as a synthesized SSE stream (see `nemo_gym/responses_streaming.py`).

Each request gets a fresh `CODEX_HOME` with a generated `config.toml` that pins a Gym-owned model provider (no `codex login` needed — the `openai_api_key` config value is handed to the subprocess as `OPENAI_API_KEY`, the provider's `env_key`), sets `approval_policy = "never"`, and disables everything that would make a rollout depend on ambient host state or phone home: analytics, update checks, on-disk history, server-side web search, and the multi-agent tool. Session persistence is disabled via `--ephemeral`. The `CODEX_HOME` and scratch working directory are removed after the run, so rollouts cannot contaminate one another.

For legacy callers, Codex is auto-installed lazily on the first local invocation via npm or a local Node.js binary if not already on PATH.

## Configuration

```yaml
codex_agent:
  responses_api_agents:
    codex_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_verifier
      concurrency: 32
      model: null
      openai_api_key: ${openai_api_key}
      openai_base_url: null
      sandbox_mode: danger-full-access
      timeout: 600
      system_prompt: null
      reasoning_effort: null
      codex_version: 0.144.4
      cwd: null
      stream_idle_timeout_ms: null
      extra_config: {}
```

- `concurrency`: max simultaneous `run()` calls
- `model`: model name written into the generated config. `null` uses the Codex CLI's own default; Gym model servers substitute their configured model anyway, so this mainly matters for direct endpoints
- `openai_api_key`: API key for the endpoint, or any non-empty string for local endpoints
- `openai_base_url`: if set, used as the provider `base_url` (include `/v1`; Codex appends `/responses`). Leave null for the real OpenAI API
- `sandbox_mode`: Codex sandbox policy for model-generated shell commands (`read-only`, `workspace-write`, `danger-full-access`). The default is `danger-full-access` because Gym environments are expected to provide their own isolation (mirroring the Claude Code agent's skip-permissions default); OS-level sandboxing (Landlock/seccomp) is unavailable in many containers
- `timeout`: per-request wall-clock seconds
- `model_context_window`: optional served context window for Codex's context accounting
- `model_auto_compact_token_limit`: optional compaction threshold, at most 90% of an explicit context window
- `system_prompt`: inserted as a `developer` role message via Codex's `developer_instructions` config. The data's system message (if any) is appended after this
- `reasoning_effort`: passed as `model_reasoning_effort` (e.g. `low`, `medium`, `high`)
- `codex_version`: **required** — npm version pinned on auto-install. Every config must pin an explicit version so runs are reproducible and cannot silently drift as new Codex releases land; version bumps become explicit, tested changes
- `cwd`: working root handed to `codex exec --cd`. `null` creates a fresh temp dir per request and removes it afterwards
- `stream_idle_timeout_ms`: provider stream idle budget. Gym model servers emit the synthesized SSE only once the full response is computed, so this must cover an entire generation; `null` defaults it to `timeout * 1000`
- `extra_config`: extra `config.toml` content deep-merged over the generated base config — add MCP servers, feature flags, `model_verbosity`, etc. Per-rollout Gym MCP entries take precedence on name collisions

For the full set of Codex config options see the [Codex configuration reference](https://developers.openai.com/codex/config).

## Gym MCP tools

When the resources server exposes Gym-owned MCP tools (an `MCPResourcesServer` returning MCP metadata from `/seed_session`), the agent writes a per-rollout `mcp_servers` entry into the generated config.toml: a streamable HTTP server pointing at the resources server's `/mcp` endpoint, with the per-rollout session token carried on a custom header via `http_headers`. Codex advertises these tools to the model as a `namespace` tool spec; the Gym model server flattens them to plain `<namespace>__<tool>` functions on the way in and splits the names back on the way out, so third-party models can call them.

## Skills evaluation

Skills are evaluated as a run-level variable, not a dataset field — point `skills.path` at a directory of [Agent Skills standard](https://agentskills.io/specification) skill directories on `gym eval run`, and the agent stages them into each request's `CODEX_HOME/skills/`, where Codex's native skill discovery picks them up:

```bash
gym eval run --agent reasoning_gym_codex_agent \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output rollouts_variant_a.jsonl \
    +skills.path=skills/variant_a/
```

Each rollout result is stamped with a `skills_ref` for provenance and grouping during reward profiling, exactly as for the Claude Code agent (see its README for the full workflow).

## Legacy direct-agent limitations

- Eval only for now. Token IDs and logprobs are not wired up yet.
- Token counts come from Codex's own usage reporting (`turn.completed`).
- `turns_used` counts assistant messages right now, not tool calls.
- Codex has no `--max-turns` equivalent; runaway rollouts are bounded by `timeout`.
- Multi-turn dataset inputs are collapsed to a single prompt: only the first `system` message (as `developer_instructions`) and the last `user` message are passed to `codex exec`; any earlier user/assistant/tool turns in `responses_create_params.input` are dropped. This matches the Claude Code agent and is fine for single-turn datasets like reasoning_gym, but datasets that encode prior conversation turns in `input` will not see that history.

Without `sandbox_access`, configure `sandbox_provider` and `sandbox_config`
(`SandboxSpec` fields such as `image`, `workdir`, and `ttl_s`) on the agent.
It creates a sandbox, runs there, and destroys it on close; the default workdir is
`/app`. A supplied access always wins, including its workdir; connection failure
never triggers a replacement or host execution. With neither access nor a usable
provider, setup fails. Verifiers that inspect task files must keep using a
Resources-owned sandbox, since agent-owned sandboxes are gone before verification.
