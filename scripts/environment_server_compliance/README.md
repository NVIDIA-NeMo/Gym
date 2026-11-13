# Environment Server compliance

A non-blocking CI check that flags agent servers that do not follow the Environment Server design. It checks the
agents a PR changes, and every agent when the suite itself changes or on a manual dispatch of `cicd-main.yml`.
Findings are warnings in the PR and are summarized in the check run **Environment Server compliance (non-blocking)**.

## Rules

| Rule | Flags |
| --- | --- |
| A1 | An agent that defines `run()`. |
| A2 | An agent that does not implement session seed and close. |
| A3 | An agent that overrides `aggregate_metrics`. |
| A4 | An agent whose seed fails when it is lent a sandbox, other than by refusing it with a 422 that says `does not support sandbox access`. |
| A5 | An agent that stops a borrowed sandbox when its session closes. |

## Run locally

```bash
python -m scripts.environment_server_compliance.ci check responses_api_agents/<agent>
```

This installs the agent's venv the way Gym does for its tests and prints one warning per finding.

## Layout

| File | Role |
| --- | --- |
| `engine.py` | Rule types and evaluation. Standard library only. |
| `rules.py` | The rules. |
| `component.py` | The agent's default config, entry point and server classes. |
| `probe.py` | Seeds and closes a session on a sandbox lent to the agent, for A4 and A5. |
| `inspection.py` | Runs in the agent's venv and writes the findings. |
| `ci.py` | Selects the changed agents and checks one. |
| `checks.py` | Publishes the check run. |

A rule is a `check(target)` that returns the offending function or class, or None, plus the ids of the rules it
`requires`.
