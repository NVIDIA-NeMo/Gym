# Claude Code Agent

Runs Claude Code CLI (`claude -p`) as a NeMo Gym agent server.

Docs: [Anthropic Messages](https://docs.nvidia.com/nemo/gym/main/model-server/anthropic-messages) — how every Gym model server exposes `/v1/messages`, and how to wire this agent to any `model_server`.

## Quick start

### env.yaml

For Anthropic API:

```yaml
anthropic_api_key: sk-ant-...
anthropic_model_name: claude-sonnet-4-6
anthropic_base_url: null
```

For a local vLLM or Ollama endpoint that serves the Anthropic Messages API:

```yaml
anthropic_api_key: EMPTY
anthropic_model_name: Qwen/Qwen3-4B-Instruct-2507
anthropic_base_url: http://localhost:8000
```

`anthropic_base_url` should not include `/v1`. Claude Code appends `/v1/messages` itself.

### Launch

For a quick eval against Anthropic's API (or any endpoint set via `anthropic_base_url`), pass the resources server config, which includes the agent server config:

```bash
gym env start --resources-server reasoning_gym/reasoning_gym_claude_code_agent
```

#### Against a Gym model server

Every Gym model server now exposes `POST /v1/messages` (a default Messages ↔ Responses mapping on `SimpleResponsesAPIModel`), so Claude Code can run against any backend Gym serves — vLLM, OpenAI, an inference provider. Set the agent's `model_server` ref to that server (it takes precedence over `anthropic_base_url`); the harness resolves `ANTHROPIC_BASE_URL` to it and the CLI appends `/v1/messages`.

`reasoning_gym_claude_code_agent_model_server.yaml` wires the agent's `model_server` ref to `policy_model`. Compose it with any model server (here a vLLM serving `policy_model`):

```bash
gym env start \
  --resources-server reasoning_gym/reasoning_gym_claude_code_agent_model_server \
  --model-type vllm_model
```

This path needs only the model server's `policy_base_url`, `policy_api_key`, and `policy_model_name` (in `env.yaml` or as `+` overrides) — no `anthropic_*` vars.

Use `vllm_model` for OpenAI-compatible **chat** endpoints (vLLM, NVIDIA build, most providers) — it forwards to `/chat/completions`. `openai_model` forwards to the OpenAI **Responses** API (`/responses`), which only OpenAI/Azure implement, so it 404s against chat-only providers.

### Run the agent

```bash
gym eval run --no-serve \
    --agent reasoning_gym_claude_code_agent \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output claude_code_rollout.jsonl \
    --limit 1
```

For the model-server config above, use `--agent reasoning_gym_claude_code_agent_model_server`.

### Smoke test

Check the `/v1/messages` proxy and the real-CLI seam without a full rollout. Launch a model server, then take its URL from the `gym env start` log (`'url': 'http://127.0.0.1:<port>'`):

```bash
gym env start --model-type vllm_model \
  +policy_base_url=https://integrate.api.nvidia.com/v1 \
  '+policy_api_key=${oc.env:NVIDIA_API_KEY}' +policy_model_name=meta/llama-3.1-8b-instruct

# 1. proxy speaks Anthropic Messages (add "stream": true for the SSE path):
curl $URL/v1/messages -H 'content-type: application/json' \
  -d '{"model":"x","max_tokens":64,"messages":[{"role":"user","content":"2+2?"}]}'

# 2. the real Claude Code CLI runs against it:
ANTHROPIC_BASE_URL=$URL ANTHROPIC_AUTH_TOKEN=local \
  claude -p --output-format stream-json --max-turns 2 --model meta/llama-3.1-8b-instruct -- "What is 2+2?"
```

## Description

The agent runs `claude -p` as an async subprocess for each request. Claude Code handles all tool execution (Bash, file read/write) internally. The agent parses the stream-json output into NeMoGym output items and forwards the response to a resources server for verification.

Claude Code talks to the model via the Anthropic Messages API (`/v1/messages`). This means it can connect to Anthropic's API directly, to any local endpoint that implements `/v1/messages` (vLLM, Ollama), or — via the agent's `model_server` ref — to any NeMo Gym model server, since every Gym model server now serves `/v1/messages` by mapping Messages ↔ Responses around its own `responses()` backend.

By default the agent runs with `--bare`, which skips auto-discovery of hooks, skills, plugins, MCP servers, memory, and CLAUDE.md so each scripted call starts clean and fast; Claude still has access to Bash, file read, and file edit tools. This isolation is the default because it keeps evals reproducible — a rollout depends only on the model, the task input, and the explicit config, not on ambient state of the host. This is the recommended mode for scripted and SDK calls per [Claude docs](https://code.claude.com/docs/en/headless#start-faster-with-bare-mode). The runtime is configurable via `bare`, `mcp_config`, and `settings` (see [Runtime capabilities](#runtime-capabilities)).

For local execution, Claude Code is auto-installed on the first local invocation
via npm or a local Node.js binary if not already on PATH.

## Configuration

```yaml
claude_code_agent:
  responses_api_agents:
    claude_code_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_verifier
      concurrency: 32
      model: claude-sonnet-4-6
      anthropic_api_key: ${anthropic_api_key}
      anthropic_base_url: null
      max_turns: 30
      timeout: 300
      system_prompt: null
      allowed_tools: null
      disallowed_tools: null
      claude_code_version: null
      thinking: null
      max_thinking_tokens: null
      fold_thinking: null
      bare: true
      mcp_config: null
      settings: null
```

- `concurrency`: max simultaneous `run()` calls
- `model`: model name. Full names like `Qwen/Qwen3-4B-Instruct-2507` are kept as-is for local endpoints; the provider prefix is stripped only when `anthropic_base_url` is not set
- `anthropic_api_key`: Anthropic API key, or any non-empty string for local endpoints
- `anthropic_base_url`: if set, used as `ANTHROPIC_BASE_URL`. Leave null for the real Anthropic API
- `max_turns`: passed to `--max-turns`. Set to `null` to omit the flag entirely (unlimited turns)
- `timeout`: per-request wall-clock seconds. On timeout the CLI is killed and the transcript streamed so far is returned
- `system_prompt`: appended to Claude Code's built-in system prompt via `--append-system-prompt`. The data's system message (if any) is also appended after this.
- `allowed_tools`: passed to `--allowedTools` (e.g. `"Bash,Read"`)
- `disallowed_tools`: passed to `--disallowedTools`
- `claude_code_version`: npm version to pin on auto-install (null means latest)
- `thinking`: passed to `--thinking` (`disabled`, `adaptive`, or `enabled`)
- `max_thinking_tokens`: passed to `--max-thinking-tokens` to cap thinking token usage
- `fold_thinking`: whether thinking is folded into the assistant message text as `<think>...</think>`. `null` (default) folds it unless training-token capture is on. `false` returns the capture-style transcript without capture: thinking as separate `reasoning` items, so the final message holds only the answer (useful when a judge grades the final message). `true` is rejected under capture
- `bare`: when `true` (default), pass `--bare` to skip auto-discovery of hooks, skills, plugins, MCP servers, memory, and CLAUDE.md. Set to `false` to let Claude Code discover those from `CLAUDE_CONFIG_DIR` and the working directory
- `mcp_config`: path to an MCP server config file, passed to `--mcp-config`. Explicit, so it works regardless of `bare`
- `settings`: path to a settings JSON layered into the per-run `CLAUDE_CONFIG_DIR/settings.json`. Top-level keys override the defaults; the `env` block is shallow-merged so telemetry stays disabled unless you override it

For the full set of Claude Code CLI flags see the [CLI reference](https://code.claude.com/docs/en/cli-reference).

## Runtime capabilities

The agent defaults to a plain `bare` CLI call for simplicity and reproducibility. Use the `bare`, `mcp_config`, and `settings` knobs (documented above) to opt into other common setups:

- **Skip auto-discovery (default):** `bare: true`, `mcp_config: null`, `settings: null`.
- **Enable auto-discovery:** set `bare: false`. Claude Code then auto-discovers from `CLAUDE_CONFIG_DIR` and the working directory.
- **Add MCP servers:** set `mcp_config` to a config file path. `--mcp-config` is explicit, so it applies even with `bare: true`.
- **Layer custom settings:** set `settings` to a JSON file path. It is merged into the per-run `CLAUDE_CONFIG_DIR/settings.json` (env shallow-merged onto the telemetry-disabling defaults).

The per-run `CLAUDE_CONFIG_DIR` is created fresh for each request and removed afterward, so opted-in content is staged per rollout and does not leak between runs. This is the staging seam reused by skills evaluation (placing skills under `CLAUDE_CONFIG_DIR/skills/`).

## Skills evaluation

Skills are evaluated as a run-level variable, not a dataset field — the same skill-agnostic dataset is reused across skill variants (mirroring how `prompt_config` works). You point `skills.path` at a directory of [Agent Skills standard](https://agentskills.io/specification) skill directories on `gym eval run`, and the agent stages them into each request's `CLAUDE_CONFIG_DIR/skills/` so Claude Code's native discovery picks them up. When skills are present, `--bare` is forced off for that request regardless of the `bare` config.

Expected layout (each skill is a directory with a `SKILL.md`):

```
skills/variant_a/
├── cot_enhanced/
│   └── SKILL.md
├── tool_focused/
│   ├── SKILL.md
│   └── references/
│       └── api_spec.md
└── baseline/
    └── SKILL.md
```

Compare two variants over the same dataset by changing only `skills.path`:

```bash
gym eval run --agent reasoning_gym_claude_code_agent \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output rollouts_variant_a.jsonl \
    +skills.path=skills/variant_a/

gym eval run --agent reasoning_gym_claude_code_agent \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output rollouts_variant_b.jsonl \
    +skills.path=skills/variant_b/
```

Each rollout result is stamped with a `skills_ref` for provenance and grouping during reward profiling:

```json
{
  "reward": 1.0,
  "skills_ref": {
    "path": "skills/variant_a/",
    "hash": "a1b2c3…",
    "skills": [{"name": "cot_enhanced", "description": "..."}]
  }
}
```

`hash` is a content digest of the skill directory, so optimizer loops (e.g. ACE, GEPA, EvoSkill) that mutate a skill **in place** at the same path still produce distinguishable variants. For concurrent candidate evaluation, give each candidate its own directory (`skills/cand-0/`, `skills/cand-1/`, …) to avoid a path-reuse read/write race.

The skills path is resolved like `input_jsonl_fpath` (relative paths check the working directory, then the Gym root). For distributed runs the directory must be on storage accessible to the agent process.

## Limitations

- With `model_server`, model calls go through Gym and can be captured. Direct Anthropic or
  `anthropic_base_url` runs bypass Gym capture.
- Enable training-token capture with run-level `token_id_capture.enabled: true` and either the
  agent's `token_id_capture: true` or run-level `token_id_capture.all_agents: true`.
- With training-token capture on, thinking is returned as separate `reasoning` items instead of
  `<think>` text. The response `id` is the last non-empty served model response id from the main
  conversation; subagent frames are excluded from this transcript. Pending tool calls remain as
  `function_call` items with `status="incomplete"` when the CLI times out. These witnesses let Gym
  attribute the parent transcript without assigning its reward to a subagent's captured chain.
  The served id is not a declared terminal id on `/run`; capture still corroborates it against the
  transcript. Without capture, thinking stays folded and unmatched tool calls remain omitted, unless
  `fold_thinking: false` selects the capture-style transcript.
- `turns_used` counts assistant messages, not tool calls.
- Runs set `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`, so Bash has no `run_in_background` option and every
  command runs in the foreground under the Bash tool's timeout. The CLI waits for background tasks before it
  exits, so a backgrounded command would hold the run open after the final answer.
- Runs set `CLAUDE_CODE_MAX_RETRIES=0`, so the CLI sends each model request once. Claude Code 2.1.108 would
  resend a failed request ten times, up to 32 seconds apart (160 to 200 seconds in all), and it resends a 401
  and a 409 like a 5xx; no setting leaves those out. The Model Server still retries its own upstream failures,
  and the Environment Server retries the episode after a 408, 425, 429 or 5xx. One native resend remains: after
  a 404 when the stream is created, or a stream that fails midway, the CLI sends the request once more without
  streaming.

## Sandbox execution

The same `claude_code_agent` supports standard agent sessions. An EnvironmentServer
seeds `/v1/agent_sessions`, sends one text activation to the rollout-scoped
`/v1/responses`, and closes `/v1/agent_sessions/close` to collect observations.
Retries of the same activation reuse its result; a different activation requires a
new session. Unseeded requests retain the existing local behavior.

For an agent-owned sandbox, add these fields to the agent config:

```yaml
claude_code_version: "2.1.108"
sandbox_provider: sandbox  # top-level provider config, shared with other harnesses
sandbox_config:
  image: ubuntu:24.04
  workdir: /app
sandbox_install_timeout_seconds: 600
session_close_timeout_seconds: 60
```

The image must include coreutils, `gzip`, `tar`, and the libraries the Claude binary
requires; it needs no Python. Seed installs a missing `bash` on a root image with `apk` or
`apt-get`, and otherwise fails with a message that names it. Seed probes the sandbox's architecture and libc, downloads
and caches the matching Linux x64/arm64 release on the **Gym host**, then compresses
and uploads it in bounded pieces through the sandbox provider's file transfer API.
The sandbox needs no internet access for runtime setup. The Gym host needs HTTPS
access to the release service, or a supplied local binary as described below.

The sandbox installer decompresses the transfer, verifies the native bytes, and
atomically installs them in `/tmp/nemo-gym-claude-runtime-<version>`, separate from
session state. A file lock serializes installation. Later seeds in the **same
sandbox** verify the cached checksum and version before skipping the binary
upload; a fresh sandbox still needs the transfer. Closing a borrowed session
removes its private state and retains the shared runtime for subsequent sessions.

Host downloads use the current user's owned, mode-0700
`~/.cache/nemo_gym/claude` directory and are checked against the release manifest's
SHA-256. Verified file identities are memoized in each worker; edits or replacements
trigger verification again. Transfers stream from the compressed cache with at
most four 1 MiB pieces staged or uploading per worker, shared across sessions.
For an independently trusted checksum, set
`sandbox_executable_sha256`. An offline Gym host can supply
`sandbox_local_executable_path` pointing to a matching Linux binary on the host;
alternatively, `sandbox_executable_path` names a binary already **inside the
sandbox**. These two options are mutually exclusive and require
`sandbox_executable_sha256`.

`network_access: model_only` limits an agent-owned sandbox's egress to the model
endpoint's host. The CLI is uploaded from the Agent Server, so setup needs no other
host. The OpenSandbox policy is set when the sandbox is created, so a session that
borrows a task sandbox rejects any value other than `inherit` (the default, the
provider's network). Only the OpenSandbox provider enforces the policy, so every
other provider rejects a restricted value, including ones with network controls of
their own. `network_access` cannot be combined with
`sandbox_config.provider_options.network_policy`. Claude uses no tool grants, so
`model_and_tools` is rejected. Each of these refusals fails the seed with HTTP 422.
Under `model_only` the image must already include `bash`: the step that installs a
missing one on a root image runs after the sandbox starts, when the package mirrors
are out of reach. A borrowed sandbox relies on its owner's network policy. The policy
is only as strong as the OpenSandbox server's egress mode: in `dns` mode only name
resolution is filtered, so a hard-coded IP bypasses it; `dns+nft` also enforces it at
the IP layer and is needed for full isolation.

With neither `model_server` nor `anthropic_base_url` set, the CLI calls Anthropic's
API directly, so `model_only` allows `api.anthropic.com`, and the real API key is in
the sandbox. This path has not been run live behind the deny policy. It stops fetches
by host name, but that host also serves Anthropic's server-side web search, so the
policy alone may not stop the model from searching: pair it with
`disallowed_tools: WebSearch,WebFetch`. With a Gym model server, the sandbox holds
only placeholder credentials and this does not arise; the benchmark profile uses one.

Configure `model_server` as above, or `anthropic_base_url` and `anthropic_api_key`;
the endpoint must be reachable from the sandbox. With `model_server`, only
placeholder Anthropic credentials enter the sandbox; the configured direct
Anthropic credential stays on the host. `resources_server` is needed only
for the legacy agent `/run` path. Sandbox-only servers install no host CLI. For
local execution, installation happens on the first local invocation; concurrent
calls share setup, failures propagate, and a later call can retry.

The default YAML selects composition with an Environment Server rather than acting as an unbound
legacy swap source. To use `--resources-server X --agent-type claude_code_agent`
with legacy swap configurations, explicitly restore
`resources_server: {type: resources_servers, name: "???"}` on the agent. Existing
resource-bound recipes retain their bindings.

For borrowed execution, the seed request supplies `sandbox_access` with a direct
provider connection descriptor and an absolute task `workdir`. Resources prepares
the task directory; Claude Code only stages its own runtime and settings outside
it. Required remote tool accesses are rejected: this adapter runs Claude Code's
native tools. Optional static `mcp_config` and `settings` JSON files are staged
from the agent server, with the same command options as local execution. A
`skills.path` in the merged server config is copied into the session's
`CLAUDE_CONFIG_DIR/skills`, including supporting files; skills disable `--bare`
through the same command builder used locally. This fallback is sandbox-only:
local `/run` uses only its explicit `skills_ref`, and unseeded `/v1/responses` does
not pick up configured skills. The collector's
run-level skills option remains unsupported for materialized task rows.

Sandbox requests accept one user text prompt (a string or a single user message),
optionally preceded by one system message. Assistant/developer messages, multiple
user turns, and request `instructions` are rejected with HTTP 422 so history or
instructions cannot be silently dropped. Use the leading system message or
`system_prompt` for instructions. One session runs one activation and does not
resume a previous Claude conversation. Local prompt conversion is unchanged.

Sandbox requests may set `metadata.agent_timeout_sec` to a positive number of
seconds, encoded as a string. This bounds the activation's model and tool work,
capped by the configured `timeout`; missing or invalid values use that ceiling.
Other metadata keys are rejected with HTTP 422. Setup and cleanup keep their
separate limits.

The shared `SandboxSession` supervisor stops the harness before reading its
dedicated stdout stream and native transcripts. The runner keeps `events.jsonl`,
`stderr.log`, and `runtime.json` separate from supervisor diagnostics. Close returns the observation bundle, disconnects
borrowed sandboxes, and stops agent-owned sandboxes. The task directory remains
available to the borrowed sandbox's owner for verification. `timeout` bounds
harness execution; setup and close use the limits above. The close response's
`cleanup_confirmed` reports whether harness termination was witnessed or an owned
sandbox was stopped. A false value can accompany successful borrowed-connection
release; the owner must destroy that sandbox and leave the activation ungraded.

`configs/profiles/benchmark.yaml` is a swappable profile for benchmarks whose Resources server supplies no sandbox,
such as HLE. It leaves `resources_server.name` unset (`???`), so Gym composes this agent onto the benchmark's own
agent instance, and each session creates its own sandbox through the `sandbox` provider. That sandbox must reach the
model server by address, so the profile sets `use_absolute_ip: true` and keeps the head server and the agent itself
on `127.0.0.1`: the head server's config endpoint is unauthenticated, and an unseeded request to the agent runs the
harness on the Gym host. It also points the CLI at the `policy_model` Gym model server, which serves the Anthropic
Messages API, with an empty `anthropic_api_key`, and pins `claude_code_version`, which sandbox sessions require. The
profile sets the sandbox's lifetime (`ttl_s`), which OpenSandbox otherwise ends after 10 minutes. Supply the image
in `sandbox_config.image`; set other `sandbox_config` keys under the profile's own instance name,
`claude_code_agent.responses_api_agents.claude_code_agent.sandbox_config`.
It sets `network_access: model_only`, so the sandbox reaches only the model server; the image must
include `bash`.

HLE, HLE Diamond, GPQA and APEX Shortlist need no Environment Server setting: their judges serve no tools, so their
Environment Servers grant the agent none. This agent rejects a required `direct_http` grant with a 422, so a
benchmark whose Resources server serves its tools over direct HTTP does not pair with it.
