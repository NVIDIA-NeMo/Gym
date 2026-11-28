# AutomationBench

AutomationBench scored with the Artificial Analysis headline metric by default.

## Recipes

| Config | Version | Agent | Path |
| --- | --- | --- | --- |
| `config.yaml` | 0.2.0 | `simple_agent` | `single_agent_turn` Environment Server with the [`automationbench` resources server](../../resources_servers/automationbench/README.md) |
| `nooa.yaml` | none | `nooa_agent` (embedded) | `single_agent_turn_legacy` Environment Server, direct HTTP tools |
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

## Other harnesses

`nooa.yaml` runs `nooa_agent` in its embedded mode on top of `config.yaml`: the same resources server, rows,
sampling and `aa_headline` scoring, and the api tools over the same direct HTTP grant. It has no manifest and is
not an AA-comparable run: NOOA's own CodeAct loop replaces `simple_agent`'s tool loop, and the step limit changes
from `max_steps: 49` tool rounds to `max_policy_calls: 50` model calls (verifiers' `max_turns`). The composed config
resolves; no rollout was run, so whether NOOA honors the run-level `temperature` and `top_p` and the rows'
131072-token output budget is unchecked.

Hermes, OpenCode, Codex and Claude Code have no recipe, because the config cannot supply what they need:

- Hermes, OpenCode and Codex take tools only over MCP grants and reject the direct HTTP grant this server uses.
  The server would need `expose_tools_over_mcp: true` and an `mcp_tools` override that lists its three tools (its
  only tool route is the `/{tool_name}` catch-all, which auto-exposure does not list), and the environment
  server `resources_tool_transports: [mcp]`.
- Codex also rejects a tool list in the request, so these rows (which carry one) need a harness-neutral
  preparation, as `benchmarks/gdpval/prepare_harness_neutral.py` does.
- All three would run in their own sandbox (`sandbox_provider` and an `image` in `sandbox_config`, as in each
  agent's `profiles/benchmark.yaml`), which this task does not otherwise need.
- Claude Code takes no tool grants at all, so it cannot reach the world's tools without an adapter change.

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
