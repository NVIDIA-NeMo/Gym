# Codex Agent

Runs the OpenAI Codex CLI (`codex exec`) as a NeMo Gym agent server.

## Quick start

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

Codex is auto-installed on first startup via npm or a local Node.js binary if not already on PATH.

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

## Limitations

- Eval only for now. Token IDs and logprobs are not wired up yet.
- Token counts come from Codex's own usage reporting (`turn.completed`).
- `turns_used` counts assistant messages right now, not tool calls.
- Codex has no `--max-turns` equivalent; runaway rollouts are bounded by `timeout`.
- Multi-turn dataset inputs are collapsed to a single prompt: only the first `system` message (as `developer_instructions`) and the last `user` message are passed to `codex exec`; any earlier user/assistant/tool turns in `responses_create_params.input` are dropped. This matches the Claude Code agent and is fine for single-turn datasets like reasoning_gym, but datasets that encode prior conversation turns in `input` will not see that history.
