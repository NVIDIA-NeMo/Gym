# Codex Agent

Runs the [Codex](https://github.com/openai/codex) CLI (`codex exec --json`).

Meant to be customized if needed and currently eval-only. 

## Quick start

codex must be on PATH (auto-installed on first start, or `npm install -g @openai/codex`). Set
`policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

Codex only speaks the OpenAI Responses API and streams, so it routes through `vllm_model`, whose
`/v1/responses` accepts streaming and converts responses->chat (so the chat tool-call parser yields
structured tool calls). `model_server: policy_model` makes the agent point at it.

```bash
ng_run "+config_paths=[resources_servers/math_with_judge/configs/math_with_judge_codex_agent.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts +agent_name=math_with_judge_codex_agent \
  +input_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl \
  +output_jsonl_fpath=codex_rollout.jsonl +limit=5
```

Per request the agent writes a `config.toml` (provider with `wire_api = "responses"`, base_url
resolved from `model_server`, `[features] multi_agent = false`) into an isolated `CODEX_HOME`, runs
one `codex exec` with stdin from `/dev/null`, then parses the `--json` events: `agent_message` ->
message, `command_execution` -> `function_call` + `function_call_output`, `turn.completed` -> usage.

## Routing notes

- Codex requires a streaming `/v1/responses` that accepts `function` tools. The Gym `vllm_model`
  provides this (it buffers the responses->chat result and re-emits Responses SSE), so codex routes
  through the model server (the training-ready path). The backing vLLM must serve chat with a tool
  parser (e.g. `--tool-call-parser hermes --enable-auto-tool-choice`).
- `multi_agent = false` is required: codex's `multi_agent` namespace tool is rejected by non-OpenAI
  `/v1/responses` schemas.
- To go direct instead, drop `model_server` and set `base_url` to an endpoint whose `/v1/responses`
  streams and accepts function tools (a real OpenAI endpoint does; the NVIDIA endpoint rejects
  function tools on `/v1/responses`).

## Config fields

- `model_server`: Gym model server to route through (preferred); its URL becomes the provider base_url
- `concurrency`: max simultaneous `run()` calls
- `command`: the codex command, split on spaces
- `model`: model id (the model server may override it with its configured model)
- `model_provider`: provider id written to config.toml
- `base_url`: provider base url, used only when `model_server` is unset
- `api_key_env`: env var name codex reads the key from
- `wire_api`: `responses` (codex requirement)
- `env`: extra env vars for the subprocess (e.g. the api key)
- `workspace_root`: where per-request workspaces are created and deleted
- `system_prompt`: prepended to the user message
- `timeout`: seconds for the `codex exec` run
- `extra_args`: extra flags appended to `codex exec`
- `codex_version`: npm version to pin on install (null means latest)
