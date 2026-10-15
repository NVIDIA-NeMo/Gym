# AutomationBench

AutomationBench scored with the Artificial Analysis headline metric by default.

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

Off by default. To keep every assertion's verdict in the rollout, collect with

```bash
++automationbench.responses_api_agents.verifiers_agent.export_state_columns='[assertion_results]'
```

Each rollout then carries
`response.exported_state.assertion_results`, one record per assertion in task
order: `index`, `type`, `app` (the WorldState service the assertion inspects),
`role` (`objective`, `guardrail` or `unscored`), `passed` (final-world verdict;
a guardrail with `passed: false` is a violation) and `initially_passed`
(initial-world verdict; `null` for unscored assertions and for tasks without an
initial state). The records use the same classification as the count metrics.
If the rubric could not score a rollout (an assertion raised), the key is
omitted and the agent logs a warning.

## Install

The env package is not on PyPI, install it into the agent's venv for data prep.
Running environment will create its own venv.

```bash
uv pip install -e benchmarks/automationbench
```

This pulls `automation-bench` from the upstream repo. Note upstream declares
`requires-python >=3.13`.

## Data

```bash
python benchmarks/automationbench/prepare.py
```

Pulls the public taskset from the installed `automation-bench` package (600 tasks: 6 domains x 100).
