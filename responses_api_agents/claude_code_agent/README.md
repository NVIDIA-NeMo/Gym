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

No model server is needed for basic eval. To extend this agent to training, a model server should be developed that handles messages endpoint. For evals with the current version, just pass the resources server config, which includes the agent server config, as is the current standard in NeMo Gym:

```bash
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym_claude_code_agent.yaml]"
```

### Run the agent

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_claude_code_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=claude_code_rollout.jsonl \
    +limit=1
```

## Description

The agent runs `claude -p` as an async subprocess for each request. Claude Code handles all tool execution (Bash, file read/write) internally. The agent parses the stream-json output into NeMoGym output items and forwards the response to a resources server for verification.

Claude Code talks to the model via the Anthropic Messages API (`/v1/messages`). This means it can connect to Anthropic's API directly, or to any local endpoint that implements `/v1/messages` such as vLLM or Ollama. It does not go through a Gym model server, but that is the next step to extend this integration to training and additional features.

The agent runs with `--bare`, which skips auto-discovery of hooks, skills, plugins, MCP servers, auto memory, and CLAUDE.md so each scripted call starts clean and fast; Claude still has access to Bash, file read, and file edit tools. To enable MCP servers or skills, remove `--bare` and add the relevant flags in `app.py`'s `_run_claude_code` command list.

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

For the full set of Claude Code CLI flags see the [CLI reference](https://code.claude.com/docs/cli-reference).

## Limitations

- Eval only for now. Token IDs and logprobs are not wired up yet.
- Does not go through Gym's model server. Token counts come from Claude Code's own usage reporting.
- `turns_used` counts assistant messages right now, not tool calls. 

