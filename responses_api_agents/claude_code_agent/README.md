# Claude Code Agent

Runs Claude Code CLI (`claude -p`) as a NeMo Gym agent server.

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
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym_claude_code_agent_model_server.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
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

For the model-server config above, use `+agent_name=reasoning_gym_claude_code_agent_model_server`.

### Smoke test

Check the `/v1/messages` proxy and the real-CLI seam without a full rollout. Launch a model server, then take its URL from the `ng_run` log (`'url': 'http://127.0.0.1:<port>'`):

```bash
ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml]" \
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

Claude Code is auto-installed on first startup via npm or a local Node.js binary if not already on PATH.

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
      bare: true
      mcp_config: null
      settings: null
```

- `concurrency`: max simultaneous `run()` calls
- `model`: model name. Full names like `Qwen/Qwen3-4B-Instruct-2507` are kept as-is for local endpoints; the provider prefix is stripped only when `anthropic_base_url` is not set
- `anthropic_api_key`: Anthropic API key, or any non-empty string for local endpoints
- `anthropic_base_url`: if set, used as `ANTHROPIC_BASE_URL`. Leave null for the real Anthropic API
- `max_turns`: passed to `--max-turns`
- `timeout`: per-request wall-clock seconds
- `system_prompt`: appended to Claude Code's built-in system prompt via `--append-system-prompt`. The data's system message (if any) is also appended after this.
- `allowed_tools`: passed to `--allowedTools` (e.g. `"Bash,Read"`)
- `disallowed_tools`: passed to `--disallowedTools`
- `claude_code_version`: npm version to pin on auto-install (null means latest)
- `thinking`: passed to `--thinking` (`disabled`, `adaptive`, or `enabled`)
- `max_thinking_tokens`: passed to `--max-thinking-tokens` to cap thinking token usage
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

Skills are evaluated as a run-level variable, not a dataset field — the same skill-agnostic dataset is reused across skill variants (mirroring how `prompt_config` works). You point `skills.path` at a directory of [Agent Skills standard](https://agentskills.io/specification) skill directories on `ng_collect_rollouts`, and the agent stages them into each request's `CLAUDE_CONFIG_DIR/skills/` so Claude Code's native discovery picks them up. When skills are present, `--bare` is forced off for that request regardless of the `bare` config.

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
ng_collect_rollouts +agent_name=reasoning_gym_claude_code_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=rollouts_variant_a.jsonl \
    +skills.path=skills/variant_a/

ng_collect_rollouts +agent_name=reasoning_gym_claude_code_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=rollouts_variant_b.jsonl \
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

- Eval only for now. Token IDs and logprobs are not wired up yet.
- Does not go through Gym's model server. Token counts come from Claude Code's own usage reporting.
- `turns_used` counts assistant messages right now, not tool calls.
