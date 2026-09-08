# ab_verified

AutomationBench-Verified: a cleaned task set with the `api` toolset.

- Source: https://github.com/xeophon/AutomationBench-Verified @ 6173254 (verifiers 0.3.1)
- Toolset: `api` (direct tool access, no meta-tools)
- Scoring: upstream Verified rubric

## Data

```bash
python environments/ab_verified/prepare.py
```

600 tasks, 6 domains x 100. `ab_aa` and `ab_zapier` score the same task set.
