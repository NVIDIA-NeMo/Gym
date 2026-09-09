# ab_aa

AutomationBench scored with the Artificial Analysis headline metric, so results
are comparable to the public AA leaderboard.

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
- Scoring code: `automationbench_aa_gated_env/` (imports upstream
  `automationbench`; vendors nothing)

## Install

The env package is not on PyPI; install it into the agent's venv:

```bash
uv pip install -e environments/ab_aa
```

This pulls `automation-bench` from the upstream repo. Note upstream declares
`requires-python >=3.13`.

## Data

```bash
python environments/ab_aa/prepare.py
```

600 tasks, 6 domains x 100. `ab_zapier` and `ab_verified` score the same task set.

## Note on comparability

AA evaluates 657 tasks with 22,822 objectives (~34.7 per task); this env uses
the public 600-task set at roughly 9.5 objectives per rollout. The metric is the
same, the split is not, so absolute numbers will not match the leaderboard.
