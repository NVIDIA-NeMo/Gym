# gym_v_smoke

This is a small adapter qualification set, **not** a reproduction of a published
leaderboard or a held-out generalization claim. It is experimental and unbaselined.

## Measurement contract

- Provider: [gym_v pinned source](https://github.com/rohitrango/gym-v/tree/7ccf14ecb5241f37372dd530accf77f7e7ef5fc6). The adapter uses the native environment's
  reset, observation and action interfaces.
- Scope: FrozenLake, GameOfLife, DoorKey, GridComponent, TreeEvenPartitioning; 25 tasks, five seeds per fixture environment, one repeat.
- Seeds: 900000 + 100 × fixture index + offset (offset 0–4). These are disjoint
  from the committed examples. Disjointness from all historical training rows
  has not been audited; do not describe them as held-out training evaluation.
- Inputs: native rendered observations and rules. The agent constructs the first
  prompt after reset, so `prompt_source: agent` is intentional. Exact prompts and
  action grammar come from `environments/gym_v/config.yaml` and the example rows.
- Budgets: 1024 output tokens for FrozenLake/DoorKey; 2048 for the three single-turn puzzles, temperature 0.7. Per-task horizons are inherited from
  the fixtures; an additional agent backstop is 64 calls. No episode wall-clock
  deadline is configured. The model context limit must be specified by the run.
- History: this benchmark config defaults to full history. The adapter also supports
  optional first-plus-latest-two reasoning/image compaction through
  [Gym #3368](https://github.com/NVIDIA-NeMo/Gym/pull/3368); this branch depends on
  that shared API. See [the agent configuration](../../responses_api_agents/gymv_agent/README.md#context-compaction)
  for capture, measured-budget and model-template requirements. The separate
  compaction qualification uses a different budget and is not this fixed benchmark.
- Score: `reward_mode: task_success` returns audited binary success (0 or 1).
  Report mean success and the number of valid tasks, grouped by environment.
  Missing sessions or absent success criteria are masked infrastructure/scoring
  failures, not legitimate zero-score attempts. Model mistakes remain unmasked.
- Training compatibility: default `reward_mode: training` still returns native
  accumulated reward. The benchmark explicitly opts into binary evaluation.
- Grading path: resources-server `/verify`, observed by Gym's shared verifier route.
  Representative live grading and local failure-handling checks passed; this does
  not establish every provider's scoring correctness.
- Dataset licensing: the manifest describes our synthetic seed declaration as
  Apache-2.0; it does not license or clear redistribution of provider dependencies.

## Prepare and validate

From the repository root with the development environment installed:

```bash
python benchmarks/gym_v_smoke/prepare.py
gym env validate gym_v_smoke
```

The generated benchmark JSONL is ignored by Git. Preparation is deterministic and
uses checked-in minimal fixtures, not a remotely changing dataset.

## Integration qualification and scope

The adapter has isolated package locks, lifecycle/error-handling tests, manifest
validation, and an exported `VERIFIER_FIXTURE` exercising the public grading
path. Its synthetic completed-session cases cover binary success, positive
partial credit with task failure, unavailable state, malformed requests, and
identical verification retries. They test score handling, not every provider's
success predicate.

Before review, retain representative real-model rollout evidence showing actions,
simulator interaction and grading. A high success rate is not an integration gate.
For thinking models, record any larger output allowance as a distinct diagnostic
protocol; do not combine its scores with this fixed smoke configuration.

This contribution uses the custom agent loop through `legacy_agent`, with full
history by default and optional shared context compaction. It does not claim all-provider
qualification, a published leaderboard reproduction, training-framework recovery,
checkpoint/resume, or offline re-judging. Those are follow-up capabilities, not
prerequisites for this scoped integration. `verified: false` remains until
maintainer baselining and review.
