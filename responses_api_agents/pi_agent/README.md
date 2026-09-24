# Pi Agent

Runs the [pi](https://github.com/earendil-works/pi) CLI (`pi --print --mode json --no-session`). 
pi runs its own tools internally. Resources server for verifier.

This adapter supports evaluation; token IDs and logprobs are not captured.

## Quick start

pi must be on PATH (auto-installed on first start, or `npm install -g @earendil-works/pi-coding-agent`).
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
- `output_token_policy`: `fixed` (native behavior) or `remaining_context`; the latter
  omits both output-limit fields from Gym-provider chat-completion requests so vLLM
  can calculate the available budget. Server-side defaults may still cap output.
- `auto_compaction`: Pi automatic compaction, including overflow recovery (default true);
  the benchmark preset disables it so history is preserved until the context is full.
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `workspace_root`: where per-request HOMEs are created and deleted
- `thinking`: passed to `--thinking` (off, minimal, low, medium, high, xhigh)
- `system_prompt`: appended via `--append-system-prompt`
- `timeout`: seconds for the `pi` run
- `bash_timeout`: optional per-call Bash limit in seconds. Supplies omitted deadlines
  and caps model-requested deadlines, preserving shorter ones. Pi terminates the
  command's process tree and returns a tool error so the agent can continue.
  Unset preserves Pi's native behavior (no default Bash deadline).
- `extra_args`: extra flags appended to the `pi` command
- `models_config`: written to `~/.pi/agent/models.json`
- `pi_version`: npm version to pin on install (null means latest)
- `mcp_servers`: Gym MCP endpoints, keyed by server name, with `url`, session `headers`,
  `enabled` (default true), and `timeout` in milliseconds (default 60000).
  The dedicated `pi_sandboxed_agent` fills these from `tool_servers` for each rollout.

The bundled extension discovers Gym-hosted tools and registers them as
`<server>_<tool>` alongside Pi's native tools. It supports Gym's authenticated,
stateless JSON MCP endpoint; arbitrary MCP transports and SSE are not supported.
Connection details are written to a private file in the temporary workspace and
removed with that workspace. Initialization failures stop the rollout; tool-call
failures appear as Pi tool errors. Calls have deadlines and honor cancellation.

See `configs/pi_agent.yaml`.
