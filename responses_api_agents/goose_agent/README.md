# Goose Agent

Runs [Goose](https://github.com/aaif-goose/goose) (Apache-2.0) headless with `goose run` inside a
provider-managed sandbox. Gym stays on the host: it provisions the sandbox, routes model requests through the
Gym model server (with token capture), seeds authenticated MCP tools, exports the Goose session store, and calls
the resources server's verifier.

Goose's native subagents come from its built-in `summon` extension: the `delegate` tool runs a subagent in its own
session, either synchronously or with `async: true` followed by `load(task_id)` to collect the result. Several
delegate calls in one turn run concurrently. Delegates cannot delegate further (depth 1), and they inherit the
parent's extensions, so they see the Gym MCP tools too.

## Install

Each sandbox installs the pinned Goose 1.54.0 static (musl) release for its architecture (`uname -m`: `x86_64` or
`aarch64`) and verifies its sha256 before running it. By default the sandbox downloads the archive from the
GitHub release, which needs `curl` or `wget`, `sha256sum` and `tar` in the image and internet access. For
sandboxes without internet (`network_access: model_only` or `model_and_tools`), download both archives once to
a host directory and set `goose_archive_dir`. Gym then uploads the matching archive into each sandbox:

```bash
mkdir -p /path/to/goose-archives && cd /path/to/goose-archives
for arch in x86_64 aarch64; do
  curl -fLO https://github.com/aaif-goose/goose/releases/download/v1.54.0/goose-$arch-unknown-linux-musl.tar.gz
done
sha256sum goose-*.tar.gz  # must match GOOSE_ARCHIVE_SHA256 in app.py
```

## Configuration

See [`configs/goose_agent.yaml`](configs/goose_agent.yaml).

| Key | Meaning |
|---|---|
| `provider`, `model` | Goose provider (`GOOSE_PROVIDER`, default `openai`) and the request model name (`GOOSE_MODEL`). |
| `base_url` | OpenAI-compatible base URL ending in `/v1`. `null` routes through the Gym model server, with the rollout prefix that token capture needs. |
| `subagent_model` | `GOOSE_SUBAGENT_MODEL` for delegates. `null` reuses `model`. Through the Gym vLLM model server the request name does not change which model serves it, so a different child model needs its own endpoint behind `base_url`. |
| `delegate` | `false` disables the `summon` extension (`delegate` and `load`) for a solo control. |
| `max_turns` | `goose run --max-turns` for the root. Delegates use Goose's own limit (`GOOSE_SUBAGENT_MAX_TURNS`, default 25), settable through `env`. |
| `goose_mode` | `GOOSE_MODE`. Use `auto` (default) headless, since `approve` and `smart_approve` ask a human to confirm tool calls. Delegates always run in `auto`. |
| `env` | Extra Goose settings, for example `GOOSE_CONTEXT_LIMIT`, `GOOSE_SUBAGENT_MAX_TURNS`, `GOOSE_MAX_BACKGROUND_TASKS` (concurrent `async` delegates, default 5). |
| `tool_servers` | Resources servers whose MCP tools are attached as Goose `streamable_http` extensions. |
| `network_access` | `inherit`, `model_only` or `model_and_tools` (OpenSandbox egress allowlist, see the [OpenCode and Pi guide](../opencode_sandboxed_agent/README.md#offline-scientific-evaluation-with-opencode-or-pi)). |

The request's system and developer messages are appended to Goose's system prompt (`--system`) after
`system_prompt`. The user messages form the task. Request `temperature` is passed as `GOOSE_TEMPERATURE`.

## Outputs

The response output is the root session converted to Responses items (reasoning, messages, function calls and
their outputs). Usage sums every model call of the root and its delegates. Each rollout writes `stdout.jsonl`
(`goose run --output-format stream-json`), `stderr.log`, the Goose session store (`sessions/sessions.db`, root and
child sessions linked by `parent_session_id`) and `generation.json` under `artifacts_dir`. The verify response adds:

| Field | Meaning |
|---|---|
| `goose_subagents` | Number of delegate sessions. |
| `goose_max_concurrent_subagents` | Most delegates active at once (first to last message, second resolution). |
| `goose_failed`, `goose_exit_code`, `goose_error_type` | Execution outcome. A run fails on a non-zero exit, a timeout (`timeout`), or a model error or `max_turns` that Goose itself ends with exit 0 (`model_error`, `max_turns`). With `execution_failure_reward_zero`, a failed run is graded as an empty response, except in a resource-owned sandbox, which is graded in place. |

Sandbox or transport faults while Goose runs, and extensions that fail to start, fail the request instead of scoring it, so it can be retried.

## Run

Define one agent instance per arm on top of `goose_agent`, for example a solo control on math and a delegate arm
whose MCP tools come from a tool server:

```yaml
use_absolute_ip: true
config_paths:
- responses_api_models/vllm_model/configs/vllm_model.yaml
- nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
- responses_api_agents/goose_agent/configs/goose_agent.yaml
- resources_servers/math_with_judge/configs/math_with_judge.yaml
- resources_servers/example_mcp_weather/configs/example_mcp_weather.yaml

goose_solo:
  _inherit_from: goose_agent
  responses_api_agents:
    goose_agent:
      resources_server: {type: resources_servers, name: math_with_judge}
      delegate: false

goose_swarm:
  _inherit_from: goose_agent
  responses_api_agents:
    goose_agent:
      resources_server: {type: resources_servers, name: example_mcp_weather}
      tool_servers: [{type: resources_servers, name: example_mcp_weather}]
      delegate: true
      network_access: model_and_tools
      goose_archive_dir: /path/to/goose-archives

math_with_judge_simple_agent: null
math_with_judge_environment_server: null
example_mcp_weather_claude_code_agent: null
example_mcp_weather_environment_server: null
```

```bash
gym env start --config goose_arms.yaml
gym eval run --no-serve --agent goose_swarm --input <rows.jsonl> --output results/goose_swarm.jsonl
```

Restricted network access requires OpenSandbox and `use_absolute_ip: true`.

## Known limitations

- Goose replays earlier reasoning as `reasoning_content`, which vLLM 0.16 and later ignore when rendering the
  prompt. With token capture enabled, a multi-turn rollout therefore cannot link its calls into one chain and is
  masked. Single-call rollouts rebuild cleanly.
- The solo control removes `delegate` and `load`, but Goose's default extension manager stays enabled, so a model
  could in principle re-enable `summon` at run time.
- Delegates started with `async: true` are cancelled if the root ends its run without collecting them with `load`.
