# ab_verified

> **Not runnable on `main` yet.** AutomationBench-Verified requires the v1
> taskset support in `verifiers_agent` (the `verifiers.taskset` config path).
> `main`'s agent only accepts `vf_env_id` + `vf_env_args`, so this config fails
> validation until the v1 agent lands. It is kept here so the three
> AutomationBench variants live together.

AutomationBench-Verified: a cleaned task set using the `api` toolset.

- Source: https://github.com/xeophon/AutomationBench-Verified @ 6173254 (verifiers 0.3.1)
- Toolset: `api`
- Scoring: upstream Verified rubric

## Data

```bash
python environments/ab_verified/prepare.py
```

600 tasks, 6 domains x 100 — the same set `ab_aa` and `ab_zapier` score.
