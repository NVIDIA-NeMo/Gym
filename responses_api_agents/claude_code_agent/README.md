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
      capture_trajectory: true
```

- `concurrency`: max simultaneous `run()` calls
- `model`: model name. Full names like `Qwen/Qwen3-4B-Instruct-2507` are kept as-is for local endpoints; the provider prefix is stripped only when `anthropic_base_url` is not set
- `anthropic_api_key`: Anthropic API key, or any non-empty string for local endpoints
- `anthropic_base_url`: if set, used as `ANTHROPIC_BASE_URL`. Leave null for the real Anthropic API
- `max_turns`: passed to `--max-turns`. Set to `null` to omit the flag entirely (unlimited turns)
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
- `capture_trajectory`: when `true` (default), each `run()` result carries a standardized `trajectory` built from the session transcript Claude Code writes on disk (see [Trajectory capture](#trajectory-capture))

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

## Trajectory capture

Claude Code persists a complete session transcript — one JSON record per event, with timestamps, request ids, per-model-call token usage, and tool execution metadata — under `$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-id>.jsonl`. Since each rollout runs with an ephemeral `CLAUDE_CONFIG_DIR`, the agent harvests those artifacts just before cleanup and attaches a standardized, versioned trajectory to every `run()` result (and therefore to every rollout JSONL row). This addresses the standardized trajectory-telemetry requirements of [NVIDIA-NeMo/Gym#1867](https://github.com/NVIDIA-NeMo/Gym/issues/1867). When no transcript is available, the trajectory is built from the stream-json stdout events instead (`source: "stream_json"`), which carry the same message structure and token usage but no timestamps or request ids — missing telemetry is `null`, never fabricated.

The schema lives in `trajectory.py` (`ClaudeCodeTrajectory`, `schema_version: "1.0"`, validated via `validate_trajectory()`):

```json
{
  "schema_version": "1.0",
  "agent": "claude_code_agent",
  "source": "transcript",
  "session_id": "…",
  "model": "claude-sonnet-4-6",
  "num_turns": 4,
  "duration_ms": 8123.0,
  "total_cost_usd": 0.021,
  "result_usage": {"input_tokens": 63, "cache_read_input_tokens": 18478, "output_tokens": 912},
  "stats": {"prompt_tokens": 63, "completion_tokens": 912, "total_tokens": 975, "cached_tokens": 18478, "cache_creation_tokens": 8289, "reasoning_tokens": null},
  "steps": [
    {"step_id": 0, "type": "user_message", "timestamp": "…", "content": "fix the bug"},
    {
      "step_id": 1, "type": "agent_turn", "turn_no": 1, "timestamp": "…",
      "model": "claude-sonnet-4-6", "request_id": "req_…", "message_id": "msg_…", "stop_reason": "tool_use",
      "content": "…", "reasoning_content": "…",
      "stats": {"prompt_tokens": 12, "completion_tokens": 300, "cached_tokens": 9000, "cache_creation_tokens": 512, "total_tokens": 312, "reasoning_tokens": null},
      "tool_calls": [{"call_id": "toolu_…", "name": "Bash", "arguments": "{\"command\": \"ls\"}"}],
      "observations": [
        {"source_call_id": "toolu_…", "content": "…model-visible output…", "status": "completed",
         "started_at": "…", "completed_at": "…", "duration_ms": 532.3, "extra": {"interrupted": false}}
      ]
    }
  ]
}
```

Semantics:

- **Delta / append-only steps**: each step holds only the new content it introduced — a user message, or one complete agent turn (one model call) with its tool calls and observations. The model-visible input of turn *N* is the concatenation of all steps before it, starting from the most recent `context_boundary` step. Full request payloads are never re-materialized per turn, so the artifact stays linear in conversation size.
- **Compaction**: when Claude Code compacts its context, a `context_boundary` step is emitted whose `content` is the summary that replaced the prior history; post-compaction context = boundary content + later steps.
- **Per-model-call stats**: `stats` on each `agent_turn` carries prompt/completion/total tokens plus `cached_tokens` (cache reads) and `cache_creation_tokens`, deduplicated per API message (the transcript writes one record per content block, repeating the usage). `reasoning_tokens` is `null` — the Anthropic Messages API doesn't report it separately.
- **Tool observations and timing**: each observation records the model-visible output, `completed`/`error` status, and independent `started_at`/`completed_at`/`duration_ms` from the actual execution boundary (issuing assistant record → its tool-result record), so parallel tool calls keep independent timing. Short scalar execution metadata from Claude Code's `toolUseResult` (flags, exit info) is kept in `extra`; oversized payloads are dropped.
- **Failures**: tool errors surface as `status: "error"` observations; on a rollout timeout the partial transcript written before the kill is still harvested.
- **Subagents**: sidechain (subagent) records are out of scope for the schema and skipped, but counted in `sidechain_records_skipped` so consumers can tell "no subagents ran" from "subagent events were dropped".

Identity: `session_id` identifies the Claude Code session; task and rollout identity are recorded by Gym's rollout collection layer on the surrounding rollout row (the trajectory rides on the verify response), and each model call is identified by `request_id`/`message_id`, each tool call by `call_id`.

## Limitations

- Eval only for now. Token IDs and logprobs are not wired up yet.
- Does not go through Gym's model server. Token counts come from Claude Code's own usage reporting.
- `turns_used` counts assistant messages right now, not tool calls.
