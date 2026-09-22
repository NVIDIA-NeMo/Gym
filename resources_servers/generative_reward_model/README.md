# Generative Reward Model Resources Server

A resources server for **training** a Generative Reward Model (GenRM) with RL. Given an evaluation
prompt containing conversation context and two candidate responses, the policy under training acts
as the judge: it emits per-rubric and overall scores plus a relative ranking. This server parses that
verdict and rewards it against human ground truth.

This is the training counterpart to `resources_servers/genrm_compare/`, which *uses* an already
trained GenRM to score candidates during RLHF. Here the GenRM itself is the model being optimized.

## Overview

1. The policy receives a judging prompt and emits a single JSON verdict.
2. `score_parser.extract_scores` parses the verdict (see format below).
3. `/verify` computes an L1 penalty against the ground-truth scores and ranking.
4. The reward is the negative total penalty — `0.0` is the best achievable reward.

### Expected model output format

Exactly one JSON dict (fenced or bare) must appear in the output:

```json
{
    "rubric_evaluations": [
        {
            "rubric_id": 1,
            "response_1_analysis": "...",
            "response_2_analysis": "...",
            "score_1": 4,
            "score_2": 3,
            "ranking": 2
        }
    ],
    "overall": {
        "response_1_analysis": "...",
        "response_2_analysis": "...",
        "score_1": 4,
        "score_2": 3,
        "ranking": 2
    }
}
```

- `score_1` / `score_2`: individual helpfulness, 1–5, higher is better.
- `ranking`: 1–6, where 1 = response 1 much better, 6 = response 2 much better. The midpoint is 3.5.

**More than one JSON dict is a parse failure**, not a lenient "take the first". A degenerate policy
that repeats its verdict block would otherwise keep parsing successfully, and RL reinforces the
repetition for free. This collapsed a training run at roughly step 600: entropy fell, generations
looped the verdict JSON until they hit the token cap, and reward never moved.

## Reward

```
overall_penalty = score_weight * (|s1_err| + |s2_err|) + ranking_weight * |rank_err|
rubric_penalty  = rubric_weight * mean_over_rubrics(
                      score_weight * (|s1_err| + |s2_err|) + ranking_weight * |rank_err|
                  )
reward          = -(overall_penalty + rubric_penalty)
```

Only fields actually present in the ground truth are penalized, so a sample carrying just
`ranking` contributes only the ranking term. If the output cannot be parsed — or the predicted
rubric IDs do not exactly match the ground-truth rubric IDs — the reward is
`parse_failure_penalty`.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_weight` | float | `1.0` | Multiplier on individual score L1 error |
| `ranking_weight` | float | `2.0` | Multiplier on ranking L1 error |
| `rubric_weight` | float | `0.5` | Multiplier on the averaged per-rubric penalty (`0.5` = half the weight of the overall term) |
| `parse_failure_penalty` | float | `-100.0` | Flat reward when the output cannot be parsed |

## Data format

Each line of the dataset JSONL:

```json
{
    "id": "sample-0",
    "responses_create_params": {"input": [{"role": "user", "content": "<judging prompt>"}]},
    "ground_truth_overall": {"score_1": 4.0, "score_2": 3.0, "ranking": 2.0},
    "ground_truth_rubric_scores": [
        {"rubric_id": 1, "score_1": 4.0, "score_2": 3.0, "ranking": 2.0}
    ]
}
```

`ground_truth_overall` may contain any subset of `score_1`, `score_2`, `ranking`; missing keys are
skipped when computing the reward. `data/example.jsonl` holds five worked samples spanning the
ranking scale in both directions (overall rankings 1, 2, 3, 5, 6) with one, two, three and five
rubrics. Rubric IDs need not start at 1 or be contiguous — the server matches predicted rubrics to
ground truth by ID, and requires the two sets to match exactly.

## Reported metrics

`/verify` returns numeric fields that rollout collection averages for logging:

| Metric | Description |
|--------|-------------|
| `format_correct` | `1.0` if the verdict parsed, else `0.0` |
| `overall_score_1_l1`, `overall_score_2_l1` | L1 error on the overall individual scores |
| `overall_ranking_l1` | L1 error on the overall ranking |
| `overall_ranking_binary_acc` | `1.0` if the predicted ranking is on the same side of 3.5 as ground truth |
| `rubric_score_l1` | Mean L1 score error across rubrics |
| `rubric_ranking_l1` | Mean L1 ranking error across rubrics |

## Quick start

Point `jsonl_fpath` for the `train` dataset at your data, then start the servers:

```bash
gym env start --resources-server generative_reward_model/genrm_train
```

The config is named `genrm_train.yaml` rather than after its directory, so it registers as the
flavor `generative_reward_model/genrm_train`. The policy model doubles as the judge, so
`generative_reward_model_simple_agent` binds the `policy_model` server rather than a separate
judge model.

## File structure

```
generative_reward_model/
├── __init__.py
├── app.py                        # Resources server: parse verdict, compute reward and metrics
├── score_parser.py               # JSON verdict parsing
├── task_data.py                  # Dataset-row schema (ground truth shape)
├── configs/
│   └── genrm_train.yaml          # Configuration
├── data/
│   ├── example.jsonl             # Example samples
│   └── example_metrics.json
├── tests/
│   ├── __init__.py
│   ├── test_app.py               # Reward and metric tests
│   └── test_score_parser.py      # Parser tests
└── README.md                     # This file
```

## Development

```bash
pytest resources_servers/generative_reward_model/tests/ -v
```

## Related components

- `resources_servers/genrm_compare/` — inference-time use of a trained GenRM for RLHF scoring
- `responses_api_models/genrm_model/` — Response API model supporting `response_1` / `response_2` roles

## License

Apache 2.0 - Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
