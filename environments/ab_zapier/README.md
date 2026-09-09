# ab_zapier

AutomationBench with the upstream Zapier scoring (`partial_credit`).

- Source: https://github.com/zapier/AutomationBench (verifiers 0.2.0)
- Toolset: `zapier` (meta-tool discovery layer)
- Scoring: partial credit; a broken guardrail counts as one failed assertion

## Install

The env package is not on PyPI; install it into the agent's venv:

```bash
uv pip install -e environments/ab_zapier
```

This pulls `automation-bench` from the upstream repo. Note upstream declares
`requires-python >=3.13`.

## Data

```bash
python environments/ab_zapier/prepare.py
```

600 tasks, 6 domains x 100 - the same set `ab_aa` scores.
