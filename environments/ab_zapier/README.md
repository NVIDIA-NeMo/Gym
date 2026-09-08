# ab_zapier

AutomationBench with the upstream Zapier scoring (`partial_credit`).

- Source: https://github.com/zapier/AutomationBench (verifiers 0.2.0)
- Toolset: `zapier` (meta-tool discovery layer)
- Scoring: partial credit; a broken guardrail counts as one failed assertion

## Data

```bash
python environments/ab_zapier/prepare.py
```

600 tasks, 6 domains x 100. `ab_aa` and `ab_verified` score the same task set.
