# Kimi Code Sandboxed Agent

Runs Moonshot's [Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) (`@moonshot-ai/kimi-code`, MIT) in a
provider-managed sandbox as one print-mode session (`kimi -p ... --output-format stream-json`). Gym stays on the host:
it allocates the sandbox, routes every model call (root and subagents) through Gym model servers, seeds MCP tool
servers, downloads the native session, and calls the resources server's verifier. Kimi Code's own subagent tools,
`Agent` and `AgentSwarm`, are available unless solo mode is selected.

```bash
gym env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config responses_api_agents/kimi_code_sandboxed_agent/configs/kimi_code_sandboxed_agent.yaml \
    --config resources_servers/mcqa/configs/mcqa.yaml \
    "+kimi_code_sandboxed_agent.responses_api_agents.kimi_code_sandboxed_agent.resources_server.name=mcqa"
```

## Installation in the sandbox

Kimi Code ships as one self-contained glibc binary per platform. With `kimi_binary_path` set, Gym uploads that
host-local file into each sandbox; this is required when the sandbox has no internet access. Download it from the
release manifest and check its digest, for example:

```bash
V=2.1.1
curl -fsSL https://code.kimi.com/kimi-code/binaries/$V/manifest.json   # lists filename and sha256 per platform
curl -fsSL -o kimi-code-linux-x64 https://code.kimi.com/kimi-code/binaries/$V/kimi-code-linux-x64
sha256sum kimi-code-linux-x64   # compare with platforms["linux-x64"].checksum
```

Without `kimi_binary_path`, the sandbox runs the official installer pinned to `kimi_version` (needs `curl` and
egress to `code.kimi.com`). Either way the run fails as an infrastructure error, not a zero score, when
`kimi --version` does not match `kimi_version`. The binary is about 190 MB, so a prebuilt image or a nearby copy saves
setup time at scale. Alpine/musl images are not supported by the release binaries.

## Configuration

| Key | Effect |
|---|---|
| `model_server` | Root model. Kimi Code talks to it with its `openai` (Chat Completions) provider |
| `subagent_model_server` | Optional second Gym model server, written as Kimi's `[secondary_model]` default for `Agent`/`AgentSwarm` children. The root may still pick `"primary"` per spawn unless `subagent_model_force: true` |
| `subagents` | `false` hides `Agent` and `AgentSwarm` (solo control) |
| `swarm_max_concurrency` | Sets `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`; unset means Kimi's native ramp (5 at once, then one more every 700 ms, no cap) |
| `max_steps_per_turn` | `[loop_control] max_steps_per_turn` (unset: unlimited) |
| `print_max_turns` | `[background] print_max_turns`: how many extra root turns background-subagent completions may trigger |
| `thinking_effort` | `[thinking] effort`; Kimi K3 accepts `low`, `high` and `max` only |
| `disabled_tools` | `[tools] disabled`; the cron tools are hidden by default because they never fire in a one-shot run |
| `permission_rules` | `[[permission.rules]]` entries, for example `{decision: deny, pattern: "Bash(rm -rf*)"}` |
| `kimi_config` | Any other `config.toml` content, deep-merged last |
| `network_access`, `tool_servers` | Same contract as the OpenCode and Pi sandboxed agents: `model_only` and `model_and_tools` apply an OpenSandbox deny-default egress policy that allows the model server(s) and, for the latter, the tool servers |
| `execution_failure_reward_zero` | Score failed or answerless runs as zero (the verifier must score an empty response as an unmasked zero) |

Permission mode: print mode always runs Kimi's `auto` ("Never Ask") policy, and `--auto`/`--yolo` cannot be combined
with `-p`. In this mode `AgentSwarm` and `Agent` run without approval; static deny rules from `permission_rules` still
apply.

System and developer messages in the task input become `$KIMI_CODE_HOME/AGENTS.md` (global instructions); user
messages are joined into the prompt. Request sampling parameters such as temperature are not forwarded: Kimi Code
reads them only for its `kimi` provider type, so set them on the model server.

## Solo versus swarm

```yaml
# solo control
subagents: false
# native subagents, children on a second server, at most 16 swarm children at once
subagents: true
subagent_model_server: {type: responses_api_models, name: lightning_model}
swarm_max_concurrency: 16
```

`AgentSwarm` launches up to 128 children from one `prompt_template` and `items` list and returns one aggregated
report; only one `AgentSwarm` call is allowed per model response. Foreground `Agent` calls block the root. A
background `Agent` (`run_in_background: true`) is still waited for: in print mode Kimi keeps the process alive and
feeds each completion back to the root as a new turn (`print_background_mode = "steer"`), so children finish before
the process exits.

## Outputs

- `response.output`: the root agent's messages, tool calls and tool results from the stream-json transcript. Thinking
  is not part of that transcript.
- `response.usage`: summed over the root and every subagent from the `usage.record` events of each
  `agents/*/wire.jsonl` in the native session.
- `kimi_num_subagents`, `kimi_max_concurrent_subagents`, `kimi_subagents`: each child's id, parent, spawning tool
  call, type, model, swarm index, start/end time (ms) and status, from the `subagent.*` events.
- `ng_agent_observations` (when rollout capture is on): one invocation for the root and one per child, linked by
  parent and spawning tool call.
- `kimi_results_dir`: `stdout.jsonl`, `stderr.log`, the native `sessions/` tree (one `wire.jsonl` per agent) and
  `generation.json`.

Every root and child model call goes to `<model server>/ng-rollout/<rollout id>/v1`, so token-ID capture covers
children too.

## Known limits

- The secondary model must be a separate Gym model server, because Gym's vLLM model server replaces the request's
  model name with its own.
- Children of a swarm share the sandbox; they edit the same working tree.
- Only the root transcript becomes `response.output`; child conversations stay in the native session files.
- Tested with Kimi Code 2.1.1. The session layout (`sessions/<workdir key>/<session>/agents/<agent>/wire.jsonl`) and
  its event names are internal to Kimi Code and may change between releases.
