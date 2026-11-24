# AutomationBench

AutomationBench scored with the Artificial Analysis headline metric by default.

## Recipes

| Config | Version | Agent | Path |
| --- | --- | --- | --- |
| `config.yaml` | 0.2.0 | `simple_agent` | `single_agent_turn` Environment Server with the [`automationbench` resources server](../../resources_servers/automationbench/README.md) |
| `config_legacy_agent.yaml` (deprecated) | 0.1.0 | `verifiers_agent` | `legacy_agent` relay |

Both run the same tasks, prompts, tools and rubric. `config.yaml` keeps each
task's world in a resources server session and runs the tool loop in
`simple_agent`. Its `max_steps` is 49, not verifiers' `max_turns` of 50:
verifiers stops before running the 50th call's tool calls, while `simple_agent`
runs a call's tool calls before it checks the cap, so both change the world in
at most 49 tool rounds. The score reads only that world.
`config_legacy_agent.yaml` hands the whole episode to verifiers and stays to
reproduce existing results; the relay will be removed after the Environment
Server migration. The two manifests carry different versions because a
different agent loop is a different measurement, even where the scores agree.

Sampling: both recipes send `temperature` and `top_p` 1.0.
`config_legacy_agent.yaml` sends them from its agent config; `config.yaml` sets
them as the run's `responses_create_params`, which applies to every row of a run
that loads it and which `++responses_create_params.temperature=...` replaces.
Rows carry neither, because `verifiers_agent` would let a row's value win over
its config. Rows do carry the tools and the 131072-token output budget; rows
prepared before the resources server existed lack the tools, and `config.yaml`
refuses them at seed, before any model call, so each fails once as an
`environment_server_failed` rollout instead of being scored. Re-run
`gym eval prepare --benchmark automationbench` to get them.

The resources server takes its tool dispatch from verifiers, as a transitional
choice for exact tool parity with `config_legacy_agent.yaml`: it builds Gym's
`AutomationBenchEnv` on a one-row placeholder dataset and calls each tool
through that environment's `update_tool_args` and `tool_map`, so the tools keep
verifiers' argument handling and Gym's `api_fetch` overrides, and the server
depends on verifiers at the same git pin as `verifiers_agent`, with its uv
constraint, override and exclude lists copied. Dropping
verifiers would mean calling upstream `automation-bench`'s tools directly and
carrying those overrides in the server.

## Metric

Artificial Analysis defines the score as:

> The Score is the share of task objectives a model completes, where any task
> with a guardrail violation scores zero.

That differs from upstream `partial_credit`, which counts a broken guardrail as
just one more failed assertion and so never zeroes a task. `aa_headline`
implements the gate; `partial_credit` is kept in the rubric at weight 0 for
reference.

An assertion already passing in the initial state (and not force-scored via
`"excluded": False`) is a guardrail; everything else is an objective. This
mirrors upstream classification exactly.

Alongside the score the env reports `guardrails_violated`, `guardrails_total`,
`objectives_passed` and `objectives_total`, which makes violations-per-task
directly comparable to the AA leaderboard column.

- Toolset: `api`, matching the AA harness
- `api_fetch` uses JSON objects instead of JSON strings for `body` and
  `params`, with `{}` for empty values.
- Scoring code: `automationbench_env/` (imports upstream
  `automationbench`; vendors nothing)

## Per-assertion results (optional)

Off by default. To keep every assertion's verdict in the rollout, collect
`config.yaml` with

```bash
++automationbench_resources_server.resources_servers.automationbench.export_assertion_results=true
```

and each rollout carries `assertion_results`. With `config_legacy_agent.yaml`,
collect with

```bash
++automationbench.responses_api_agents.verifiers_agent.export_state_columns='[assertion_results]'
```

and each rollout carries
`response.exported_state.assertion_results`. Either way it holds one record per assertion in task
order: `index`, `type`, `app` (the WorldState service the assertion inspects),
`role` (`objective`, `guardrail` or `unscored`), `passed` (final-world verdict;
a guardrail with `passed: false` is a violation) and `initially_passed`
(initial-world verdict; `null` for unscored assertions and for tasks without an
initial state). The records use the same classification as the count metrics.
If the rubric could not score a rollout (an assertion raised), the sample is
masked as `verifier_error`; under `config_legacy_agent.yaml` the key is omitted
and the agent logs a warning.

## Session harnesses

`opencode.yaml`, `openclaw.yaml`, `hermes.yaml`, `pi.yaml` and `codex.yaml` run that harness in place of
`simple_agent`, with the same resources server, tasks and `aa_headline` scoring as `config.yaml`. They have no
manifest and are not AA-comparable runs: the harness's own loop, tools and limits replace `simple_agent`'s. For
example, with Pi:

```bash
gym eval prepare --benchmark automationbench/pi
gym eval run \
    --model-type vllm_model \
    --benchmark automationbench/pi \
    --split benchmark \
    --output results/automationbench_pi.jsonl \
    --model-url <vllm_base_url> \
    --model-api-key <vllm_api_key> \
    --model <served_model_name> \
    ++policy_model.responses_api_models.vllm_model.propagate_context_overflow_errors=true \
    '++policy_model.responses_api_models.vllm_model.sampling_overrides={temperature: 1.0, top_p: 1.0}'
```

- The tools reach the harness over MCP: these harnesses refuse a direct HTTP tool grant. Each preset sets
  `expose_tools_over_mcp: true` on the resources server, which then lists `api_search`, `api_fetch` and
  `base64_encode` with the names, descriptions and schemas `config.yaml`'s rows send, and the preset's own
  environment server (`automationbench_<harness>_environment_server`, so one run can load several presets) grants
  them with `resources_tool_transports: [mcp]`. Each harness shows them to the model under its own name, such as
  `automationbench_resources_server_api_fetch` (OpenCode) or `mcp__automationbench_resources_server__api_fetch`
  (Pi). A tool's text reaches the model unchanged, as over direct HTTP.
- The harnesses refuse a tool list, an output budget and sampling in the request. The presets run the rows that
  `prepare_harness_neutral.py` writes (`automationbench-600-harness-neutral.jsonl`; `gym eval prepare --benchmark
  automationbench/<harness>`): `config.yaml`'s tasks without the tools and the output budget. The resources server
  serves those rows because the presets set `rows_carry_tools: false`; with the default, it refuses a row without
  tools at seed, MCP or not. The presets load the resources server's config, not `config.yaml`, so they set no
  run-level `responses_create_params` and leave other workloads' request fields as they are. The reverse does not
  hold: a workload that sets run-level `responses_create_params` (`config.yaml` itself, or `lmarena_v3`, for example)
  sends those fields with these presets' rows too, which the harnesses refuse, so run it separately.
- Each harness runs in a sandbox it creates through the `sandbox` provider, as set up by the agent's
  `configs/profiles/benchmark.yaml` (OpenSandbox). The task needs no files in it, so the presets use `python:3.12`;
  any Linux glibc image with `bash`, `curl`, `tar`, `gzip` and CA certificates works, for example
  `++automationbench_pi_agent.responses_api_agents.pi_agent.sandbox_config.image=python:3.13`. The sandbox calls the
  model server and the resources server's `/mcp` by address, so the presets bind both to the address sandboxes
  reach (`absolute_ip_server_types`).
- OpenCode, OpenClaw, Pi and Codex download their runtime at the start of each episode, so their sandboxes keep the
  provider's network (`network_access: inherit`). To restrict one to the model server and the MCP endpoint
  (`model_and_tools`), supply the runtime in the image, as the agent's README describes. Hermes's runtime comes from
  the agent server, so `hermes.yaml` uses `model_and_tools`.
- Each preset runs at most 32 episodes at a time: the environment server admits 32 (`max_concurrent_episodes`; the
  rest wait up to `queue_timeout_seconds`, 86,400 s, before their deadline starts) and the harness runs 32
  (`concurrency`). Change both together: an episode admitted beyond the harness's `concurrency` would wait for it
  with its deadline running.
- Set sampling on the model server, as above. These harnesses send none of their own, except Hermes, whose
  `temperature` setting defaults to 1.0. With Pi, also set `propagate_context_overflow_errors=true`, as above: it
  passes the model server's context-length error to Pi, which then compacts its context and continues; by default
  `vllm_model` returns an empty, length-capped completion and the episode ends at the context limit.
- Each preset limits a model call to 32,768 output tokens, except Codex, which sends no limit, and gives the harness a
  262,144-token window; change `context_window` (Codex `model_context_window`, Hermes `context_length`) for another
  model.
- The step limits differ from `config.yaml`'s 49 tool rounds. Hermes stops at 50 turns (`max_turns`). OpenCode,
  OpenClaw, Pi and Codex have no turn limit: a run that keeps calling tools ends at the harness's 3,600-second
  `timeout` and is scored on the world as it stands then. The harnesses also keep their own tools (a shell, file
  tools, OpenCode's subagents), which act only inside the sandbox and not on the task's world.

## Install

The preparation package is not on PyPI. Its committed project lock pins the
upstream source. `gym eval prepare` syncs this project into an isolated
preparation environment; it does not install into the agent or parent venv.

```bash
uv sync --locked --project benchmarks/automationbench
```

This pulls `automation-bench` from the upstream repo. Note upstream declares
`requires-python >=3.13`.

## Data

```bash
uv run --locked --project benchmarks/automationbench python benchmarks/automationbench/prepare.py
```

Pulls the public taskset from the installed `automation-bench` package (600 tasks: 6 domains x 100).
