# OpenAir Congestion Reviewer Evidence

This file maps the contribution's claims to code, tests, and checked-in
artifacts. It intentionally excludes historical run receipts and external
model results that a reviewer cannot reproduce from this branch.

## Intended claim

The contribution provides a deterministic, parameter-aware, multi-turn 5G
congestion-control environment. A policy reads KPI telemetry and eight bounded
tool schemas, emits exactly one tool call, and receives the next synthetic KPI
state plus a decomposed programmatic reward.

The default `replay` backend is causal within its synthetic dynamics. The
`dataset_replay` backend is non-causal and diagnostic-only. Neither backend
claims live OAI/FlexRIC actuation or physical-network fidelity.

## Review surface

The implementation is centered on `resources_servers/openair_congestion/`.
The PR also contains this narrow supporting surface:

- the OpenAir server's explicit-close route;
- Gymnasium-agent cookie continuity and opt-in explicit-close cleanup;
- corresponding Gymnasium-agent tests;
- the root environment index.

These files are review scope, not evidence of upstream acceptance.

## Correctness evidence

| Area | Checked behavior | Evidence |
|---|---|---|
| Protocol and guardrail | Exactly one known, schema-valid tool call is required; malformed turns advance as `noop` plus a negative surcharge, while topology and safety bounds remain authoritative server-side. | `tests/test_app.py`, `tests/test_guardrail.py` |
| Causal replay | The eight-tool surface has deterministic replay semantics; focused tests cover representative parameter-sensitive effects, persistent-setpoint idempotency, and the per-cell throughput-capacity invariant. | `tests/test_replay_action_semantics.py`, `tests/test_reward_correctness.py` |
| Programmatic verifier | Reward measurements and terms are computed from before/action/after state without an LLM judge; ordering and rejection costs are tested. | `openair_congestion/rewards.py`, `tests/test_reward_correctness.py`, `tests/test_reward_profiles.py` |
| Transactionality and lifecycle | A failed step does not partially commit; transport retries are deduplicated; close, truncation, protocol failure, and lease reclamation preserve state ownership. | `tests/test_replay_lifecycle.py`, `tests/test_app.py` |
| Model input | KPI messages omit evaluator-only difficulty, regime, and scenario labels. | `tests/test_render.py`, `tests/test_example_artifacts.py` |
| Recorded-data boundary | Nested KPI-snapshot JSONL inputs fail closed on malformed topology, identifiers, action metadata, or step order; optional recorded actions are returned only as diagnostics, stored scalar rewards are ignored, and the current action cannot alter a prerecorded next state; metadata reports `training_usable: false`. | `dataset_backend.py`, `tests/test_dataset_ingestion.py` |
| Transition completeness | Every successful scored step, including terminal and truncated steps, retains its after-observation. | `app.py`, `generate_example_rollouts.py`, `tests/test_app.py`, `tests/test_example_artifacts.py` |
| OpenAir lifecycle integration | The server advertises explicit close; the agent preserves caller routing/auth cookies, merges resource-issued session cookies, closes opted-in sessions, and preserves completed rollouts with a `cleanup_warning` if cleanup fails. | `tests/test_app.py`, `responses_api_agents/gymnasium_agent/tests/test_app.py` |

## Checked-in evidence and generators

- `data/example.jsonl` contains five neutral task rows spanning the synthetic
  congestion regimes without exposing their labels to the policy.
- `data/example_rollouts.jsonl` contains compact scripted records produced
  through the actual reset/step/close API. Regeneration and focused tests check
  wiring, reproducibility, reward decomposition, and bounded completion—not
  model quality.
- `data/example_metrics.json` is the NeMo Gym example-validation artifact.
- `golden_set.py` exhaustively evaluates a finite action grid at deterministic
  decision states, applies one intervention, and coasts with `noop`. Its
  derived labels are a reward-and-dynamics sanity oracle for that grid and
  horizon, not a universal multi-step optimum or real-model evidence.

## Explicit non-claims

This branch does not claim:

- live OpenAirInterface or FlexRIC measurements or actuation;
- physical-simulator fidelity;
- policy improvement from SFT, GRPO, or any hosted model;
- that recorded `dataset_replay` transitions support on-policy training;
- that the finite-grid golden action is globally optimal; or
- upstream maintainer acceptance.

Hosted-policy evaluation, reward profiling, and NeMo RL training use the
standard Gym workflows documented in the README. They are optional empirical
work, separate from the reproducible code-correctness evidence above.

## Reproduction commands

```bash
PYTHONPATH=.:resources_servers/openair_congestion \
  .venv/bin/python resources_servers/openair_congestion/generate_example_rollouts.py

PYTHONPATH=.:resources_servers/openair_congestion \
  .venv/bin/python resources_servers/openair_congestion/golden_set.py \
  --out results/openair_congestion_golden_set.jsonl

.venv/bin/pytest \
  resources_servers/openair_congestion/tests \
  responses_api_agents/gymnasium_agent/tests/test_app.py -q

.venv/bin/gym env test --resources-server openair_congestion
```
