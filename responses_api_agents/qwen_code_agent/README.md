# Qwen Code Agent

Runs the [Qwen Code](https://github.com/QwenLM/qwen-code) CLI (`@qwen-code/qwen-code`, Apache-2.0) headless
inside a provider-managed sandbox. Gym stays on the host: it creates the sandbox, routes every model call
through a Gym model server, seeds authenticated MCP tools, exports the session transcripts (root and
subagents), and calls the resources server's verifier. Qwen Code's native subagents (the `agent` tool,
nested delegation, forks, background agents) and the experimental Agent Team are available and
configurable. This harness is currently for evaluation; token IDs and logprobs are not wired up.

## Run

Build the runtime archive once. It holds a checksum-verified Node.js build for the sandbox architecture
plus Qwen Code, so sandboxes need no internet access:

```bash
responses_api_agents/qwen_code_agent/build_runtime.sh /path/to/qwen-runtime-0.25.1-x64.tar.gz 0.25.1 x64
export QWEN_CODE_RUNTIME_ARCHIVE=/path/to/qwen-runtime-0.25.1-x64.tar.gz
export OPENSANDBOX_DOMAIN=... OPENSANDBOX_API_KEY=...
```

The archive is uploaded into every sandbox (about 93 MB) and unpacked under `/tmp`; the sandbox image needs
glibc, `tar` and `gzip`. With `runtime_archive: null`, `qwen` must already be on the image's `PATH`.

Compose [configs/qwen_code_agent.yaml](configs/qwen_code_agent.yaml) with a model server, a resources
server, and an environment server that points at the agent. For example, `run.yaml`:

```yaml
config_paths:
- responses_api_models/vllm_model/configs/vllm_model.yaml
- resources_servers/reasoning_gym/configs/resources_only.yaml
- responses_api_agents/qwen_code_agent/configs/qwen_code_agent.yaml
qwen_code_agent:
  responses_api_agents:
    qwen_code_agent:
      resources_server: {type: resources_servers, name: reasoning_gym}
      system_prompt: Put your final answer inside <answer></answer> tags.
qwen_code_environment:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server: {type: responses_api_agents, name: qwen_code_agent}
      max_concurrent_episodes: null
      queue_timeout_seconds: null
      default_episode_timeout_seconds: 21600
      cleanup_timeout_seconds: 180
```

```bash
gym env start --config run.yaml --model-url http://<vllm-host>:8000/v1 --model <served-name>
gym eval run --no-serve --config run.yaml --agent qwen_code_agent \
  -i resources_servers/reasoning_gym/data/example.jsonl -o rollouts.jsonl --limit 5
```

The vLLM server needs tool calling enabled (`--enable-auto-tool-choice --tool-call-parser <parser>`).
Qwen Code speaks Chat Completions; Gym's model server answers its streamed requests.

## Solo and swarm settings

| Field | Default | Effect |
|---|---|---|
| `subagents` | `true` | `false` removes the `agent` and `list_agents` tools: the solo control. |
| `subagent_model` | `null` | Route id for the built-in `general-purpose` subagent. `null` inherits the root model. |
| `subagent_model_server` | `model_server` | Gym model server behind `subagent_model`, so children can run a different model. |
| `max_subagent_depth` | `null` (Qwen Code: 5) | Nesting levels; 1 keeps subagents but forbids nesting. |
| `max_parallel_agents` | `null` (Qwen Code: 10) | Concurrent background subagents. Foreground launches in one turn are not capped. |
| `agent_team` | `false` | Enables the experimental Agent Team tools (`team_create`, `send_message`, task list). |
| `max_session_turns`, `max_tool_calls` | `null` | Qwen Code's run limits (`--max-session-turns`, `--max-tool-calls`). |
| `approval_mode` | `yolo` | Qwen Code approval mode; headless runs need `yolo` to execute tools. |
| `timeout` | 3600 | Seconds for the whole run, children included. |
| `exclude_tools`, `settings`, `extra_args` | empty | Extra `--exclude-tools`, settings deep-merged into `~/.qwen/settings.json`, extra CLI flags. |

Swarm example (children on a second model server, two nesting levels, 64 background children):

```yaml
qwen_code_agent:
  responses_api_agents:
    qwen_code_agent:
      subagent_model: worker
      subagent_model_server: {type: responses_api_models, name: worker_model}
      max_subagent_depth: 2
      max_parallel_agents: 64
```

`subagent_model` works by writing a user-level `general-purpose` agent definition with that `model`. Its
system prompt is the built-in one, copied from Qwen Code 0.25.1 into
[general-purpose-agent.md](general-purpose-agent.md). Forks (`subagent_type: "fork"`) and teammates keep the
root model. The model decides whether and how much to delegate; prompts that ask for parallel work
reliably trigger it.

## Tools and network

`tool_servers` lists resources servers whose tools are exposed over MCP (`expose_tools_over_mcp: true`).
Gym seeds each one per rollout and writes it to Qwen Code's `mcpServers` with the session token header.
Subagents inherit MCP tools. Qwen Code names them `mcp__<server>__<tool>`; the resources server's verifier
normalizes those names.

`network_access` follows the OpenCode and Pi sandboxed agents: `inherit` keeps the provider's network,
`model_only` allows only the model servers, and `model_and_tools` also allows the tool servers. Restricted
modes require OpenSandbox and sandbox-reachable server addresses (`use_absolute_ip: true`).

## Headless semantics (verified with Qwen Code 0.25.1)

- A headless run waits for background subagents: it drains their completion notifications into new root
  turns and exits only when no task is unfinished. Claude Code `-p`, by contrast, drops background work.
- Nested delegation works headless up to `max_subagent_depth`. Agent Team also runs headless.
- Children's model calls go through the same rollout-prefixed Gym model-server URL as the root's.

## Outputs

`response.output` holds the root session only: reasoning, messages, tool calls and results. Qwen Code defers
MCP tools and the `agent` tool behind its `tool_search` + `tool_call` bridge; a `tool_call` is recorded as the
underlying call (for example `mcp__tools__extract_synonym_values`) so verifiers see the real tool name. Reasoning that
the model server inlines as `<think>` tags is split into reasoning items. `response.usage` is Qwen Code's
run total, which includes subagent calls. The verify response adds:

- `qwen_subagents`, `qwen_subagents_background`, `qwen_subagent_max_depth` (levels; 1 means no nesting),
  counted from the exported child session metadata;
- `qwen_subagent_input_tokens`, `qwen_subagent_output_tokens`, summed over child transcripts;
- `qwen_failed`, `qwen_exit_code`, `qwen_error_type`, `qwen_results_dir`.

`qwen_results_dir` (under `artifacts_dir`) holds `stdout.jsonl` (the `-o stream-json` events, children's
events included with their `parent_tool_use_id`), `stderr.log`, `sessions.tar.gz` (Qwen Code's
`~/.qwen/projects` with `chats/` and `subagents/<session>/agent-*.jsonl` plus `.meta.json`), and
`generation.json`. A nonzero exit, missing result event, or `is_error` result marks the run failed; with
`execution_failure_reward_zero` the verifier scores an empty response. Setup and export failures raise.

## Limits

- Agent Team teammates do not write session files; their prompts appear only in `stdout.jsonl`.
- Cron tools stay enabled (native default). A headless run waits for its scheduled jobs, so a recurring job
  runs until `timeout`. Add `cron_create` and `loop_wakeup` to `exclude_tools` when tasks should not
  schedule work.
- Token capture for training is not validated; children's calls carry the root's rollout id.
