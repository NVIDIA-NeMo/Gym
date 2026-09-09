# AutomationBench

- Source: https://github.com/zapier/AutomationBench (verifiers 0.2.0)
- Toolset: `api` (matches upstream's CLI default)
- Scoring: partial credit; a broken guardrail counts as one failed assertion

## Install

The env package is not on PyPI, install it into the agent's venv
for data prep (the env will create its own venv for rollout):

```bash
uv pip install -e environments/ab_zapier
```

This pulls `automation-bench` from the upstream repo. Note upstream declares
`requires-python >=3.13`.

## Data

```bash
python environments/ab_zapier/prepare.py
```

Pulls the real taskset from the installed `automation-bench` package and sizes
itself from it (600 tasks: 6 domains x 100). `ab_aa` indexes the identical set.
