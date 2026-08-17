# OpenAir congestion reviewer notes

This note maps the contribution's claims to code, tests, and checked-in
artifacts. Historical run receipts and external model results are left out
because reviewers cannot reproduce them from this branch.

## What this contribution does

`openair_congestion` is a deterministic, parameter-aware, multi-turn 5G
congestion-control environment. A policy reads KPI telemetry and eight bounded
tool schemas, emits exactly one tool call, and receives the next synthetic KPI
state with a decomposed programmatic reward.

The default `replay` backend is causal within its synthetic dynamics.
`dataset_replay` is non-causal and intended only for diagnostics. Neither
backend provides live OAI/FlexRIC actuation or claims physical-network fidelity.

## Files to review

Most of the implementation is under `resources_servers/openair_congestion/`.
The PR also changes a small supporting surface:

- the OpenAir server's explicit-close route;
- Gymnasium-agent cookie continuity and opt-in explicit-close cleanup;
- the related Gymnasium-agent tests;
- the root environment index; and
- the Fern tutorial and its navigation entry.

These files define the review scope. Their presence does not mean the
contribution has been accepted upstream.

## Correctness checks

| Area | What to check | Evidence |
|---|---|---|
| Protocol and guardrail | The policy must emit exactly one known, schema-valid tool call. A malformed turn advances the selected backend with `noop` plus a negative surcharge instead of ending early (`replay` is causal; `dataset_replay` is not); well-formed unsafe calls are rejected by topology and safety checks. | `tests/test_app.py`, `tests/test_guardrail.py` |
| Causal replay | Supported controls have deterministic, parameter-aware replay semantics. Traffic shedding fails closed: admission must remain at 100%, slice reservations must be empty, and a PRB cap must stay at or above the active-UE equal-share floor with enough headroom to reassign all displaced throughput. A persistent cap is suspended on later transitions that cannot fully reassign it. Focused tests cover these boundaries, persistent-setpoint idempotency, shared cell capacity, and recomputed KPIs. | `tests/test_replay_action_semantics.py`, `tests/test_reward_correctness.py` |
| Programmatic verifier | Reward measurements and terms are computed from the before state, action, and after state without an LLM judge. Tests cover reward ordering and rejection costs. | `openair_congestion/rewards.py`, `tests/test_reward_correctness.py`, `tests/test_reward_profiles.py` |
| Transactionality and lifecycle | A failed step does not partially commit. Transport retries are deduplicated. Normal completion, truncation, explicit close, and lease reclamation preserve state ownership. | `tests/test_replay_lifecycle.py`, `tests/test_app.py` |
| Model input | KPI messages omit evaluator-only difficulty, regime, and scenario labels. | `tests/test_render.py`, `tests/test_example_artifacts.py` |
| Recorded-data boundary | Nested KPI-snapshot JSONL fails closed on malformed topology, identifiers, action metadata, or step order. Optional recorded actions are returned as diagnostic metadata only, stored scalar rewards are ignored, and the current action cannot change a prerecorded next state. Metadata reports `training_usable: false`. | `dataset_backend.py`, `tests/test_dataset_ingestion.py` |
| Transition completeness | Every successful scored step keeps its after-observation, including terminal and truncated steps. | `app.py`, `generate_example_rollouts.py`, `tests/test_app.py`, `tests/test_example_artifacts.py` |
| OpenAir lifecycle integration | The server advertises explicit close. The agent preserves caller routing and authentication cookies, merges resource-issued session cookies, closes opted-in sessions, and keeps completed rollouts with a `cleanup_warning` when cleanup fails. | `tests/test_app.py`, `responses_api_agents/gymnasium_agent/tests/test_app.py` |

## Checked-in evidence

- `data/example.jsonl` has five neutral task rows covering the synthetic
  congestion regimes without exposing their labels to the policy.
- `data/example_rollouts.jsonl` has compact scripted records produced through
  the environment's reset, step, and close contract. Regeneration and focused tests check
  wiring, reproducibility, reward decomposition, bounded completion, and the
  shared-capacity invariant. They do not measure model quality.
- `data/example_metrics.json` is the NeMo Gym example-validation artifact.
- `golden_set.py` exhaustively evaluates a finite action grid at deterministic
  decision states, applies one intervention, and then coasts with `noop`. The
  derived labels provide a reward-and-dynamics sanity oracle for that grid and
  horizon. They are not a universal multi-step optimum or evidence from a real
  model.

## Claims this branch does not make

This branch does not claim:

- live OpenAirInterface or FlexRIC measurements or actuation;
- physical-simulator fidelity;
- policy improvement from SFT, GRPO, or any hosted model;
- that recorded `dataset_replay` transitions support on-policy training;
- that the finite-grid golden action is globally optimal; or
- upstream maintainer acceptance.

A bounded real-model rollout and manual trace inspection are still required by
the repository quality bar before merge. Hosted-policy profiling and NeMo RL
training use the standard Gym workflows in the README and Fern tutorial; they
remain separate from the reproducible code-correctness evidence above, and this
branch does not claim fresh SFT or GRPO improvement.

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
