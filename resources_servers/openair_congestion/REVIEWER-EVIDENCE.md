# OpenAir Congestion Reviewer Evidence

This file is the review map for the OpenAir congestion-control contribution.
It distinguishes code-level evidence checked into this branch from empirical
evidence that must be produced against external model endpoints.

## Intended claim

The contribution provides a deterministic, parameter-aware synthetic
environment for testing the NeMo Gym resource-server contract and controlled
policy comparisons. It does not claim live OAI/FlexRIC actuation or physical
network-simulator fidelity.

## Review scope

The contribution is centered on `resources_servers/openair_congestion/`, but
the PR is not limited to that directory. Reviewers should also inspect:

- `resources_servers/gymnasium/base.py`, `__init__.py`, and tests for the
  explicit-close request/response and endpoint used by stateful environments;
- `responses_api_agents/gymnasium_agent/app.py` and tests for resource-issued
  cookie isolation, deterministic cleanup, and preservation of completed
  rollouts when close reports a warning;
- the root `README.md` environment registration; and
- Fern tutorial navigation plus
  `environment-tutorials/openair-congestion.mdx`.

Those shared changes are narrow framework support for the stateful resource
contract, not evidence that the OpenAir environment is already accepted
upstream.

## Correctness evidence

| Area | Checked behavior | Evidence |
|---|---|---|
| Action semantics | Default replay tiers expose eight validated tools; T2 reset narrows model-facing tools to `noop` and UE-targeted `set_prb_cap` with bounds matching its runtime guardrail. Modeled parameters change deterministic transitions; identical setpoints are idempotent. | `tests/test_guardrail.py`, `tests/test_replay_action_semantics.py`, `tests/test_app.py`, `responses_api_agents/gymnasium_agent/tests/test_app.py` |
| Reward contract | Live, synthetic replay, and dataset replay use one version selector; T2 uses `openair_t2_v3` with service accounting; non-T2 paths preserve frozen V1. | `openair_congestion/reward_profiles.py`, `tests/test_reward_profiles.py`, `tests/test_dataset_ingestion.py` |
| Admission accounting | Requested, admitted, delivered, denied, and forcibly terminated service agree with emitted UE topology. | `tests/test_replay_action_semantics.py`, `tests/test_reward_profiles.py` |
| Transactionality | Failed reward computation does not partially commit a step; close and render synchronize with an in-flight step; concurrent resets cannot reap an allocation before its session is registered. | `tests/test_replay_lifecycle.py`, `tests/test_app.py` |
| Cleanup | Failed backend close stays tracked; a completed agent rollout survives `/close` failure with a structured warning. | `tests/test_app.py`, `responses_api_agents/gymnasium_agent/tests/test_app.py` |
| Model input | Difficulty and regime labels remain evaluator metadata and are absent from rendered/model-facing messages; T2 receives compact requested/admitted/cap decision state through the HTTP path. | `tests/test_render.py`, `tests/test_app.py`, `tests/test_example_artifacts.py` |
| Model ordering | Compliance requires zero failures; exploratory mode enforces an explicit failure-rate ceiling. Only common usable pairs participate, and adjacent deltas need a positive 95% bootstrap lower bound. | `tests/test_model_sweep.py` |
| Input integrity | Topology counts and identifiers agree; fractional CSV identifiers fail closed with provenance. | `tests/test_schemas.py`, `tests/test_dataset_ingestion.py` |
| Boundary hygiene | Gymnasium rollouts forward only resource-server-issued cookies, and legacy KPI failures return a stable public error without the internal exporter endpoint. | `responses_api_agents/gymnasium_agent/tests/test_app.py`, `tests/test_legacy_server_api.py` |
| Recorded-data boundary | The checked-in dataset fixture is the default diagnostic input; recorded actions do not change prerecorded next observations; backend metadata marks these rollouts non-causal and not training-usable. Effective reward configuration and T2 service accounting survive ingestion and are exposed in reset/step metadata. Reused GRPO episode IDs are partitioned by iteration and malformed step order fails closed. | `dataset_backend.py`, `tests/test_dataset_ingestion.py`, `README.md`, Fern tutorial |
| Transition completeness | Every successful scored step, including terminal and truncated steps, returns its after-observation; generated evidence therefore retains both sides of every transition. | `app.py`, `generate_example_rollouts.py`, `tests/test_app.py`, `tests/test_example_artifacts.py` |

## First-user workflow

The README and Fern tutorial now show:

- the canonical KPI fields and one model-facing measurement example;
- an exact checked-in and custom JSONL location;
- the `backend` and `dataset_path` settings and diagnostic commands;
- extension checklists for KPIs and tools;
- hosted-policy evaluation versus NeMo RL trainable-model configuration;
- checkpoint evaluation through a compatible model server; and
- the distinction between causal synthetic `replay`, diagnostic
  `dataset_replay`, and the nonexistent live OAI/FlexRIC backend.

The custom-data workflow intentionally does not claim that recorded
before/action/after rows create counterfactual transitions or a valid on-policy
GRPO environment.


## Generated evidence

- `data/example.jsonl` contains five neutral model prompts spanning distinct
  deterministic fallback regime dynamics without naming the regimes in
  model-facing content.
- `data/example_rollouts.jsonl` contains deterministic full trajectories through
  the real reset/step/close HTTP surface. Each transition records the action
  dynamics version, reward version, service accounting, reward measurements,
  reward terms, and the scored after-observation. SHA-256:
  `ed4102eb611f7dc6d26d8a07e878ce7b6c0cebb638134eadd01a90c15de06911`.
- `data/example_metrics.json` is the NeMo Gym example-validation receipt.
- `golden_set.py` derives a reproducible single-intervention oracle from the
  deterministic action grid; its labels are not produced by an LLM judge.

## Historical external validation — rerun required

These empirical jobs ran outside the repository and their full receipts are
not checked into this contribution. They predate the fallback-load,
regime-dynamics, and T2 tool-contract corrections in this review. The hashes
identify preserved historical results, but their scores are not comparable to
the current branch and cannot qualify it. Rerun both profiles before making
any current model-quality claim.

### Strict real-model profile

A strict 500-prompt × 16-response profile completed against Qwen3-1.7B and
Qwen3-8B:

- all four scripted anchors completed 8,000 episodes under the historical
  strict-order gate. The current gate uses the more defensible partial order
  `relief > noop`, `relief > random-valid`, and both
  `noop` and `random-valid > catastrophic`;
- each current anchor constraint is evaluated on paired prompt/repeat returns
  and must have a deterministic 95% bootstrap lower bound above zero;
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

## Local handoff verification — July 27, 2026

The runtime, regression tests, and generated artifact were verified at signed,
DCO-trailed commit `b9f5dd4705d10772ff3cd30ad96f4f28a73ce799`. The
following handoff commit changes documentation only; its complete Fern and
pre-commit checks were run on the staged documentation tree.

| Gate | Result |
|---|---|
| Affected OpenAir, shared Gymnasium, and agent tests | `239 passed` |
| Clean-install `gym env test --resources-server openair_congestion` | `215 passed` |
| Repository non-sandbox CI-equivalent suite | `1237 passed`, `170 deselected` |
| Repository sandbox CI-equivalent suite | `170 passed`, `1237 deselected` |
| Combined repository coverage | `97.66%` (required `96%`) |
| Generated rollout reproducibility | Two independent generations were byte-identical; committed SHA-256 is `ed4102eb611f7dc6d26d8a07e878ce7b6c0cebb638134eadd01a90c15de06911` |
| Ruff, formatting, diff hygiene, scoped pre-commit | Passed |
| Fern documentation check | `0 errors`; one expected unauthenticated redirect-check warning |

This receipt establishes local code-review handoff readiness. It does not
convert the failed historical model/GRPO qualifications into success, prove
live-network fidelity, or represent upstream maintainer acceptance.
