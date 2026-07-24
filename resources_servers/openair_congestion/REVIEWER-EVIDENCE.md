# OpenAir Congestion Reviewer Evidence

This file is the review map for the self-contained OpenAir congestion-control
resource server. It distinguishes code-level evidence checked into this branch
from empirical evidence that must be produced against external model endpoints.

## Intended claim

The contribution provides a deterministic, parameter-aware synthetic
environment for testing the NeMo Gym resource-server contract and controlled
policy comparisons. It does not claim live OAI/FlexRIC actuation or physical
network-simulator fidelity.

## Correctness evidence

| Area | Checked behavior | Evidence |
|---|---|---|
| Action semantics | All eight tools are validated; modeled parameters change deterministic transitions; scheduler, MCS, handover, and UL-power controls expose condition-dependent costs instead of unconditional relief; identical setpoints are idempotent. The PRB-cap schema advertises only the supported UE target, and runtime validation uses the UE identifiers actually present in each cell. | `tests/test_guardrail.py`, `tests/test_replay_action_semantics.py`, `tests/test_reward_correctness.py` |
| Reward contract | Live and replay use one version selector; T2 uses `openair_t2_v3` with service accounting; non-T2 paths preserve frozen V1. | `openair_congestion/reward_profiles.py`, `tests/test_reward_profiles.py` |
| Admission accounting | Requested, admitted, delivered, denied, and forcibly terminated service agree with emitted UE topology. | `tests/test_replay_action_semantics.py`, `tests/test_reward_profiles.py` |
| Transactionality | Failed reward computation does not partially commit a step; close and render synchronize with an in-flight step. | `tests/test_replay_lifecycle.py` |
| Cleanup | Failed backend close stays tracked; a completed agent rollout survives `/close` failure with a structured warning. | `tests/test_app.py`, `responses_api_agents/gymnasium_agent/tests/test_app.py` |
| Model input | Difficulty and regime labels remain evaluator metadata and are absent from rendered/model-facing messages; T2 receives compact requested/admitted/cap decision state through the HTTP path. | `tests/test_render.py`, `tests/test_app.py`, `tests/test_example_artifacts.py` |
| Model ordering | Compliance requires zero failures; exploratory mode enforces an explicit failure-rate ceiling. Only common usable pairs participate, and adjacent deltas need a positive 95% bootstrap lower bound. | `tests/test_model_sweep.py` |
| Input integrity | Topology counts and identifiers agree; fractional CSV identifiers fail closed with provenance. | `tests/test_schemas.py`, `tests/test_dataset_ingestion.py` |
| Boundary hygiene | Gymnasium rollouts forward only resource-server-issued cookies, and legacy KPI failures return a stable public error without the internal exporter endpoint. | `responses_api_agents/gymnasium_agent/tests/test_app.py`, `tests/test_legacy_server_api.py` |

## Generated evidence

- `data/example.jsonl` contains five neutral model prompts spanning the evaluator
  regimes without naming them in model-facing content.
- `data/example_rollouts.jsonl` contains deterministic full trajectories through
  the real reset/step/close HTTP surface. Each transition records the action
  dynamics version, reward version, service accounting, reward measurements,
  and reward terms.
- `data/example_metrics.json` is the NeMo Gym example-validation receipt.
- `golden_set.py` derives a reproducible single-intervention oracle from the
  deterministic action grid; its labels are not produced by an LLM judge.

## Executed external validation

These empirical jobs ran outside the repository and their full receipts are
not checked into this contribution. The hashes below identify the preserved
results without turning failed experiments into model-quality claims.

### Strict real-model profile

A strict 500-prompt × 16-response profile completed against Qwen3-1.7B and
Qwen3-8B:

- all four scripted anchors completed 8,000 episodes and preserved
  `relief > noop > random-valid > catastrophic`;
- Qwen3-1.7B completed 8,000 episodes with mean return `-2.2886`, but had
  1,157 parse failures;
- Qwen3-8B completed 8,000 episodes with mean return `-3.3175`, zero
  parse/invalid/infrastructure failures, and an 11.8% action-rejection rate;
- the official strict verdict was **NOT_EVALUABLE** because compliance permits
  zero failures; and
- on the 6,843 mutually usable pairs, the diagnostic 8B-minus-1.7B delta was
  `+0.733987` with paired 95% interval `[+0.690253, +0.777208]`.

Qwen3-8B is not a frontier model, and the paired diagnostic does not override
the failed zero-failure gate. The result receipt SHA-256 is
`0ed292b64846e54e5a802f17bc4009304b8bf0b575776b32d6e48995ca565165`.

### Bounded GRPO

A three-iteration H100 GRPO job also completed its process but failed
policy-quality qualification on one fixed five-case, 16-step T2 manifest:

| Arm | Mean return |
|---|---:|
| noop | `-7.309680` |
| scripted T2 relief | `-7.656386` |
| SFT before GRPO | `-10.165723` |
| GRPO after three iterations | `-10.521328` |

Post-GRPO evaluation had zero parse and infrastructure failures, but a 30%
runtime rejection rate and worse total return than the starting SFT policy.
The official qualification remains false. Its qualification-receipt SHA-256
is `3f475d9227bfa9eb60b1af729dad4583ca2190eb4d9d49196b081422909335f7`.

## External empirical gates

The following are intentionally **not** claimed by this branch:

- superiority of GRPO, SFT, or either tested real model over the fixed
  baselines;
- a passed real-model small-to-frontier capability profile;
- fidelity to live OAI/FlexRIC behavior; or
- upstream maintainer acceptance.

Before making a model-quality claim, run the compliance profile against the
intended real endpoints. It must have complete identical prompt/repeat coverage,
zero infrastructure/parse/invalid-call failures, and positive adjacent paired
mean deltas whose deterministic 95% bootstrap lower bounds are above zero.

## Reproduction commands

```bash
PYTHONPATH=.:resources_servers/openair_congestion \
  .venv/bin/python resources_servers/openair_congestion/generate_example_rollouts.py

PYTHONPATH=.:resources_servers/openair_congestion \
  .venv/bin/python resources_servers/openair_congestion/golden_set.py

.venv/bin/pytest \
  resources_servers/openair_congestion/tests \
  resources_servers/gymnasium/tests/test_app.py \
  responses_api_agents/gymnasium_agent/tests/test_app.py -q

.venv/bin/gym env test --resources-server openair_congestion
```
