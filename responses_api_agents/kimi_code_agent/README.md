# Kimi Code Agent

Runs Moonshot's [Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) **2.1.1** (`@moonshot-ai/kimi-code`, MIT) as
one print-mode session (`kimi -p ... --output-format stream-json`) inside a Gym task sandbox. The Environment Server
seeds the Resources and agent sessions, activates Kimi once, closes the agent, verifies the task, and closes
Resources. Kimi runs its own tools and calls Gym Model Servers for inference. Kimi's subagent tools, `Agent` and
`AgentSwarm`, are available unless `subagents: false`.

Capability level: **Evaluation**. The adapter returns the root conversation as Responses items, usage summed over the
root and every subagent, and an observation bundle with one invocation per subagent.

Model calls already use the rollout's capture route, but no training capture run has validated this harness yet.
A seed with capture on and a `subagent_model_server` is refused, since that server's calls would be recorded under
the policy rollout. The harness conformance probes are not run at the Evaluation level.

## Configure and run

[configs/kimi_code_agent.yaml](configs/kimi_code_agent.yaml) defines the agent defaults. For SWE-bench Pro, save this
composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/kimi_code_agent/configs/kimi_code_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: kimi_code_agent
      resources_tool_transports: []
```

Supply a `policy_model` Gym Model Server, `policy_model_name`, and the `sandbox` provider configuration in
`model-provider.yaml`. The task sandbox must be able to reach the Model Server.
The SWE-bench Pro pairing is undeclared (this agent is not in that Resources server's `allowed_agents`), so
`gym env start` needs `--allow-unsupported-pairing`.

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml --allow-unsupported-pairing

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent kimi_code_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

| Setting | Default | Purpose |
|---|---:|---|
| `kimi_version` | `2.1.1` | Release whose binary checksums are pinned in the adapter. Other values are refused. |
| `model` | `${policy_model_name}` | Model name Kimi sends for the root. |
| `subagent_model_server`, `subagent_model` | `null` | A second Model Server (and model name) for `Agent`/`AgentSwarm` children, Kimi's `[secondary_model]` default. `subagent_model_force: true` pins every child to it. `subagent_model` and `subagent_model_force` require `subagent_model_server`. |
| `subagents` | `true` | `false` disables `Agent` and `AgentSwarm`, for a solo control. |
| `swarm_max_concurrency` | `null` | `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`; `null` keeps Kimi's own ramp. |
| `max_steps_per_turn`, `print_max_turns` | `null` | Kimi's `loop_control.max_steps_per_turn` and `background.print_max_turns`. |
| `thinking_effort` | `null` | Kimi's `thinking.effort`. |
| `max_context_size` | `262144` | The model's context window, for Kimi's compaction. |
| `disabled_tools`, `permission_rules` | Cron and web tools, `[]` | Kimi tool and permission settings. `WebSearch` and `FetchURL` are off by default, so tasks are solved without looking answers up online; drop them from the list to turn web access on. |
| `kimi_config` | `{}` | Merged under the generated `config.toml`; the providers, models and tool settings above win. |
| `timeout` | `7200` | Seconds per activation, enforced by the sandbox's process supervisor. Task `agent_timeout_sec` metadata can shorten it. |
| `sandbox_install_timeout_seconds`, `session_close_timeout_seconds` | `900`, `60` | Setup and per-phase close bounds. |
| `sandbox_provider`, `sandbox_config` | `null`, `{}` | Agent-owned sandbox, only when Resources supplies no `SandboxAccess`. |
| `network_access` | `inherit` | Egress of an agent-owned sandbox; `model_only` allows only the model servers' host. |

## Runtime and sandbox setup

Kimi Code ships one self-contained glibc binary per platform (about 190 MB). Session setup installs the binary for
the sandbox's architecture under `/tmp/nemo-gym-kimi-runtime-2.1.1`. The agent server downloads it from
`code.kimi.com` once into its asset cache, checks the SHA-256 pinned in `sandbox.py` from the release manifest, and
uploads it; the sandbox checks the checksum again, checks the version Kimi reports, and reuses a runtime whose
checksum and version match. The sandbox needs a glibc image, bash, `sha256sum` and `timeout`, and no internet access. Alpine/musl
images and other architectures are refused at seed. Kimi's `Grep` and `Glob` use `rg` from `PATH`. Without it, Kimi
downloads ripgrep from `code.kimi.com` on each run, which fails in an offline or `model_only` sandbox, so the image
should provide `rg`. Kimi's native assets are extracted under the runtime directory, not the task's `HOME`.

Kimi runs in `SandboxAccess.workdir` through the shared `harness_runner.sh` under the process supervisor. Its config,
global `AGENTS.md`, sessions and logs live under `KIMI_CODE_HOME` in a private session directory, so tool shells keep
the task's `HOME`. `SandboxAccess.env` is forwarded, except `KIMI_*`, `OPENAI_*` and `MOONSHOT_*` variables. Close
removes only the session directory.

`network_access: model_only` limits an agent-owned sandbox's egress to the model servers' host (every port). The agent server uploads Kimi, so setup needs no
other host. The
OpenSandbox policy is set when the sandbox is created, so a session that borrows a task sandbox rejects any value
other than `inherit`, as does a provider without network policies, an explicit
`sandbox_config.provider_options.network_policy`, or a `subagent_model_server` on another host. Kimi Code uses no tool
grants, so `model_and_tools` is rejected. Each refusal fails the seed with HTTP 422.

## Requests, results and outcomes

The user message is the task, passed as `-p` (at most 128 KiB). The request's system or developer message and
`instructions` become Kimi's global `AGENTS.md`. Other request fields, such as `temperature` and `max_output_tokens`,
are refused with HTTP 422, as are sessionless requests and `/run`.

The response output is the root's chat messages from `--output-format stream-json`, converted to Responses items, with
`<think>` reasoning split off. Usage sums every `usage.record` in this run's session (picked by the session id Kimi
prints, so kimi processes the agent starts itself are not counted). Response metadata records `kimi_exit_code`,
`kimi_num_subagents` and `kimi_max_concurrent_subagents`.

| Kimi Code 2.1.1 ending | Response |
|---|---|
| Exit 0 with an assistant message | `completed` |
| Supervisor deadline or rollout stop, with or without output | `incomplete` (verified) |
| `context.overflow` or `loop.max_steps_exceeded` on stderr | `incomplete` (verified) |
| `provider.*` error with an HTTP status on stderr | `failed`, `model_request_error` with `status_code` |
| Any other nonzero exit, including a bare 137 | `failed`, `unknown` (the episode may be retried) |
| Exit 0 without an assistant message | `failed`, `agent_invalid_output` |
| Kimi never started | The activation raises |

## Limitations

- Tool timing is not reported (`harness_tool_timing_unavailable`). Every model call carries the rollout's
  `x-session-id` header, since Kimi's headers are static per provider. Child calls are therefore attributed to the
  root, and `model_call_ownership_unavailable` is reported whenever `subagents` is on.
- Required tool grants are refused at seed; optional grants are skipped and reported. A `SandboxAccess` user override
  is refused.
- One activation per session; there is no host execution path.

## Tests

```bash
python -m pytest responses_api_agents/kimi_code_agent/tests -q -o addopts=''
```

The lifecycle tests run the shared process supervisor and runner on Linux with a fake Kimi CLI.
