# Hermes Agent

Runs [Hermes](https://github.com/NousResearch/hermes-agent) inside a task sandbox through
Gym's agent-session interface. The agent server stays outside as an adapter.

## Configure and run

[configs/hermes_agent.yaml](configs/hermes_agent.yaml) is the default harness definition.
The benchmark owns task data, preparation, verification, and task sandbox settings.
The harness owns its runtime and model/tool loop. The Environment Server binds the two
and closes the agent before verification.

Run from the Gym repository root with Gym and the benchmark's preparation dependencies
installed. For SWE-bench Pro, save this composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/hermes_agent/configs/hermes_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: hermes_agent
      resources_tool_transports: []

hermes_agent:
  responses_api_agents:
    hermes_agent:
      enabled_toolsets: [terminal]
```

Supply the `policy_model` Gym Model Server, `policy_model_name`, and `sandbox` provider in
`model-provider.yaml`. The agent's `model` defaults to `${policy_model_name}` and remains
overridable. The sandbox must be able to reach the Model Server.

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent hermes_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

The collector calls Environment Server `/run`: seed Resources, seed the agent, call its
rollout-prefixed `/v1/responses`, close the agent, verify, then close Resources. Prepared
flat rows use `single_agent_turn_legacy` with this agent session lifecycle; no additional
materialization script is needed. Collection does not call the agent's compatibility `/run`.
Pass the same configuration to startup and `--no-serve` collection; collection does not
inherit routing settings from the running servers.

### Switch harness or benchmark

To change a compatible harness, replace its config import, harness-specific settings,
the Environment Server's `agent_server.name`, and the collection command's `--agent`.
Keep benchmark data, preparation, and verifier settings unchanged. To change a compatible
benchmark, replace its Resources config/reference and prepared input, keeping the harness
definition unchanged. Check tool grants, task-image/runtime support, model API, and the
benchmark's declared `allowed_agents` before running a new pairing.

Use this explicit composition for now. `--agent` selects a configured agent; it does not
install or rebind one. The existing `--agent-type` swap and automatic benchmark-data lookup
still depend on legacy Agent-to-Resources bindings; they are not equivalent to this workflow.
Existing [Hermes SWE-Pro recipes](../../benchmarks/swebench/pro/hermes.yaml) remain supported
for compatibility; a new pairing does not need another combined preset.

## Configuration example

```yaml
hermes_agent:
  responses_api_agents:
    hermes_agent:
      entrypoint: app.py
      resources_server: null
      model_server:
        type: responses_api_models
        name: policy_model
      model: ${policy_model_name}
      enabled_toolsets: [terminal, file, code_execution]
      max_turns: 30
      concurrency: 32
      temperature: 1.0
      sandbox_provider: sandbox
      sandbox_config:
        image: my-agent-image
        ttl_s: 3600
        workdir: /workspace
      system_prompt: |
        your system prompt here.
```

| field | default | description |
|-------|---------|-------------|
| `enabled_toolsets` | `null` (all) | forwarded to `AIAgent(enabled_toolsets=...)` |
| `disabled_toolsets` | `null` | forwarded to `AIAgent(disabled_toolsets=...)` |
| `model` | `${policy_model_name}` in the default YAML | served model ID; configs that omit it retain the legacy `model_server.name` fallback |
| `resources_server` | `null` | required only for the agent's compatibility `/run`; Environment Server binds Resources for agent sessions |
| `max_turns` | `30` | maps to `AIAgent.max_iterations` |
| `concurrency` | `32` | max simultaneous `run()` calls |
| `temperature` | `null` | sampling temperature passed to `AIAgent`; request `temperature` overrides it, including `0.0` |
| `terminal_backend` | `local` | sets `TERMINAL_ENV` (process-global); `local`, `docker`, `daytona`, `modal`, `ssh` |
| `terminal_timeout` | `60` | sets `TERMINAL_TIMEOUT` (process-global); per-command wall-clock seconds |
| `sandbox_provider` | `null` | named provider used to create an agent-owned sandbox when Resources does not supply `sandbox_access` |
| `sandbox_config` | `{}` | `SandboxSpec` fields used with `sandbox_provider`; ignored when Resources supplies a sandbox |
| `sandbox_runner_timeout_seconds` | `21600` | bounds one sandbox activation; the episode deadline still applies |
| `system_prompt` | `null` | joined with request `instructions` and the first input system message, in that order; appended to Hermes' built-in prompt |
| `session_close_retry_window_seconds` | `300` | session close receipt retention from successful cleanup; retries do not extend expiry |
| `checkpoints_enabled` | `false` | turns on Hermes' own checkpoints (`checkpoints.enabled` in its config); unrelated to Gym's partial-rollout checkpoints, which this agent joins as restart-only |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->

Host and sandbox execution use the same request rules. User tasks remain user messages.
Request `model` must match the configured model. Unsupported non-default controls return
HTTP 422, including `top_p` (even `1.0`), `store`, `service_tier`, and request metadata.
`max_output_tokens` also returns 422: Hermes does not enforce a total response token budget.
Config `max_tokens` limits each model call. Only text input is supported; `developer` messages
are rejected.

Compatibility: the host path previously let config `system_prompt` replace the dataset's
system message and ignored request `temperature`. It now combines the prompts and honors
the request temperature, just like sandbox execution. These changes can affect scores.
Remove unsupported fields that older versions silently ignored.

## Migrate from the standalone sandboxed agent

`hermes_sandboxed_agent` and its configuration files have been removed. Follow the
[shared sandboxed-agent migration workflow](../../resources_servers/swebench_pro/README.md#migrate-standalone-sandboxed-agents)
to run `hermes_agent` through an Environment Server. For Hermes specifically:

- Move model, toolset, and sampling settings to your agent deployment's
  `responses_api_agents.hermes_agent` configuration. The old settings fall into two groups.
  Renamed or replaced: `sandbox_timeout` becomes `sandbox_runner_timeout_seconds`, the runner
  deadline, whose shipped default is 21600 against the removed default of 2700;
  `cleanup_timeout` becomes this agent's own `session_close_timeout_seconds`, default 30 and
  used for individual cleanup operations rather than one aggregate teardown deadline,
  against the removed default of 120; and `chat_template_kwargs` becomes the narrower
  `chat_template_kwargs_enabled` boolean, so arbitrary template keyword arguments now belong
  on the Model Server. Removed with no counterpart: `runtime_python`, `remote_run_root`,
  `results_dir`, `api_timeout`, and `context_length`. These require migration rather than
  renaming. `api_timeout` is no longer an adapter setting, but the pinned Hermes runtime still
  passes an SDK request timeout of 1800 seconds by default, configurable with `HERMES_API_TIMEOUT`
  in the sandbox environment. `sandbox_runner_timeout_seconds` separately bounds the whole
  activation. `context_length` set both Hermes' main context window and the summarization
  auxiliary's; without that override, Hermes resolves the window through its runtime metadata
  and fallback logic, which can change when compaction fires on long trajectories.
  `results_dir` kept per-run artifacts on the host; update consumers
  to use Environment Server rollout and agent-observation records. A successful session close
  returns available observations or capture-gap diagnostics even when activation fails.
- The Environment Server's `default_episode_timeout_seconds` and `cleanup_timeout_seconds`
  bound the episode and each registered cleanup callback, respectively. The agent and
  Resources Server still own sandbox teardown. Note that the
  shipped `sandbox_runner_timeout_seconds` equals the default episode deadline. Without a
  smaller task `metadata.agent_timeout_sec`, the episode deadline can interrupt the runner;
  leave time for setup and verification when choosing the runner or task budget.
- The agent-specific `configs/apptainer.yaml` has been removed. Copy its `exec` timeouts into
  your shared configuration under `sandbox.apptainer`, but drop `default_binds`: it existed
  only for the `/opt/hermes` mount this change removes. Review `extra_start_args` before
  reusing it: containment can exhaust the configured session directory during runtime
  installation or hide a runtime baked into the image under `/tmp`. Validate the actual
  `/tmp` mounts and available space, using an explicit runtime bind when needed. Apptainer's
  administrator-controlled `sessiondir max size` defaults to 64 MiB; `--workdir` can supply
  containment's temporary directories. Dropping containment alone does not expose a baked
  runtime when the host's `/tmp` is mounted there.
  See the
  [Apptainer provider reference](../../nemo_gym/sandbox/providers/apptainer/README.md#selecting-and-configuring-the-provider).
- The old `prepare_runtime.sh`, libc-selecting launcher, and `/opt/hermes` mount convention
  have been removed. This agent uses the Hermes revision pinned in `requirements.txt`. That
  pin is older than the standalone adapter's and tracks a different repository: the standalone
  runtime defaulted to `NousResearch/hermes-agent` at the September 2026 `v0.21.1` release,
  while this agent pins the `cmunley1` fork at a May 2026 commit carrying the token-id,
  chat-template, and sampling patches Gym needs. Expect API-surface differences if you were
  tracking upstream. Changing the revision means editing `requirements.txt` and rebuilding the
  agent server environment; the removed `HERMES_VERSION` and `HERMES_REPO_URL` variables are
  no longer read.
- The runtime is installed at seed time, or reused from
  `/tmp/nemo-gym-hermes-runtime-<key>/venv`, where `<key>` is the first twelve characters of
  the pinned commit rather than the full hash. Network-restricted deployments must preinstall
  exactly that path, and the virtual environment must satisfy
  `python -c 'import run_agent, mcp'`. The old portable glibc/musl bundles are not a drop-in
  replacement; validate each task image's architecture and libc before migrating a batch,
  and note that the removed runtime also bundled a static `ripgrep` for images whose BusyBox
  `grep` lacks the GNU options Hermes falls back on.
- Remove request `max_output_tokens`: this agent rejects it with HTTP 422. Config
  `max_tokens` limits each model call, not the whole response, and it is configuration only:
  unlike the removed adapter, which mapped the request field onto Hermes' per-call limit, the
  budget cannot be varied per request. Retune budgets explicitly rather than treating the two
  limits as equivalent, and check `max_turns` and `terminal_timeout` in the configuration
  table above, which both default lower here than in the removed adapter.
- Update reporting that consumes standalone `hermes_*`, `verifier_reward`, or
  `agent_image_provenance` fields to use Environment Server rollout and observation records.
  SWE-Pro image provenance still reaches the rollout record through the verifier's own
  `image_provenance` fields. Failure records change shape as well: the standalone adapter
  emitted `_ng_failure_class="agent_run_error"`, while the session path emits
  `_ng_failure_class="environment_server_failed"` alongside `_ng_failure_stage` and
  `_ng_failure_terminal`, so update anything that filters on the old string. Incomplete runs,
  infrastructure failures, and scoring exclusions follow the Environment Server's outcome
  policy; compare coverage as well as rewards when migrating.

The Hermes revision, execution settings, and failure reporting differ between the adapters,
so previous benchmark scores are not an accuracy baseline for the migrated configuration.
Revalidate representative tasks, including Alpine/musl images and offline deployments when
used by your configuration.

## Runtime and model requirements

Sandbox sessions live in the memory of the worker that seeded them. With several agent workers, session routing sends each session's requests to the worker that seeded it. Calling the agent's `/run` directly keeps no session.

Each sandbox session installs the Hermes version pinned in `requirements.txt`, the same one this server runs, at seed time unless it already imports from `/tmp/nemo-gym-hermes-runtime-<commit>/venv`, for example because the image bakes it in or an earlier session in the same sandbox installed it. Installing needs outbound access to GitHub and the Python package index. Hermes calls the Model Server directly from the sandbox, so the sandbox must also reach the Model Server at its configured host and port. Its image must also match the host CPU architecture and C library because the host's `uv` executable is copied into the sandbox.

Sessions use MCP tool grants and reject other required grants. Each granted MCP server is added to that session's Hermes configuration, so the sandbox must reach it at the granted URL, usually the Resources Server's `/mcp` endpoint. An activation fails before its first model call when a required server does not connect. Hermes names MCP tools `mcp_<server>_<tool>`; the response reports them as `mcp__<server>__<tool>`, the form Gym strips before verification, while captured model calls keep Hermes' names. The session token in each grant is readable inside the sandbox and gives access only to that episode's tools.

The Hermes runner and the model's terminal tool execute as the same user in the same sandbox. The host reads the final result, including token IDs, from `/tmp/nemo-gym-hermes-sessions/<session-id>/output.json`; commands issued by the model can also write that file. Sandbox mode is suitable for evaluation, but it must not be used to produce RL training data until results are returned through a channel the model cannot modify.

## Local compatibility

Calls without an agent session run Hermes on the agent-server host. They do not operate on
a Resources-owned task sandbox. The agent's compatibility `/run` requires an explicit
`resources_server`; agent sessions and direct `/v1/responses` do not.

The default YAML now selects composition with an Environment Server rather than acting as an unbound legacy
swap source. Legacy swap configurations must explicitly restore
`resources_server: {type: resources_servers, name: "???"}` on the agent. Existing bound
recipes, including `hermes_math` and `swebench/pro/hermes`, keep their bindings.

For the existing host-side math example, put `policy_base_url`, `policy_api_key`, and
`policy_model_name` in `env.yaml`, then run:

```bash
gym env start --config environments/hermes_math/config.yaml --model-type openai_model

gym eval run --no-serve \
  --config environments/hermes_math/config.yaml --model-type openai_model \
  --agent hermes_math_agent \
  --input environments/hermes_math/data/example.jsonl \
  --output hermes_agent_rollout.jsonl --limit 1
```

Example math rollouts are in `environments/hermes_math/data/example_rollouts.jsonl`.
For the local adapter, terminal provider failures retain their response and observation evidence and reach verification. The returned rollout is always masked (`mask_sample: true`, `failure_kind: agent_request_failed`), even if the verifier awards a positive reward, so provider failures remain excluded from scores. Agent sessions report the failure cause; the Environment Server's `agent_outcome_policy` decides whether to verify, retry, or fail without grading. Worker and cleanup failures remain execution errors. Correlated model requests carry an invocation header through SDK retries, including attempts without a response ID.
