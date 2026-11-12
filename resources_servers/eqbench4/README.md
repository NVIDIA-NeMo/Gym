# EQ-Bench 4 Resources Server

Judges a policy's roleplay conversation against published EQ-Bench 4 conversations of the same scenario, and rates
the policy on the EQ-Bench 4 leaderboard scale. The [`eqbench4` benchmark](../../benchmarks/eqbench4/README.md)
describes the measurement contract, provenance and differences from upstream; this page covers the server.

## Verify

`eqbench4_agent` sends the row, the conversation (`transcript`), and how it ended (`conversation_status`):

| `conversation_status` | Result |
| --- | --- |
| `completed`, `persona_ended` | Judged against every opponent in the row; reward is the mean fraction |
| `policy_failed` | No judge calls; reward 0, masked with the `failure_kind` of its `policy_stop_reason` |
| `persona_failed` | No judge calls; reward 0, masked as `eqbench4:persona_failed` |

A stopped policy's `policy_stop_reason` maps to `failure_kind` as: `context_limit` → `model_context_limit`,
`output_budget` → `eqbench4:policy_output_budget`, `content_filter` → `eqbench4:policy_content_filter`,
`empty_reply` (also when no reason is given) → `eqbench4:policy_empty_reply`.

For each opponent the pairwise judge reads both conversations twice, policy first and opponent first, filled into the
row's `pairwise_prompt` with seeded respondent codes. Judge `n` of `judge_model_servers` reads comparison
`comparison_index` modulo the panel size. A reading's verdict is the JSON object in the reply (a fenced block, the
whole reply, or its outermost braces); a dimension counts when its value is a respondent's code followed by `+` signs.
Weighted margins from both readings are averaged; the policy's fraction is 0.5 plus or minus half the margin as a share
of the largest possible margin. A reading without a JSON verdict is asked again up to `judge_max_attempts` times in
all; then the comparison rests on the other order alone and records the failed one in `unreadable_orders`. When
neither order has a verdict, the row fails with a judge error and is masked as `judge_failed`.

When `trait_judge_model_server` is set, the trait judge scores the conversation's behavioural traits from the row's
`trait_prompt`. It starts outside the grading record and never changes the reward; a failure is kept in
`trait_judge_error`.

The response adds `comparisons` (per opponent: fraction, margins, and each judge reading with its verdicts and text),
`policy_turns`, `trait_scores`, `trait_judge_text` and `trait_judge_error`.

## Metrics

`compute_metrics` reads the published comparisons at `reference_comparisons_path` (written by the benchmark's prepare
script) and fits Bradley-Terry ratings to them together with the policy's comparisons:

| Metric | Meaning |
| --- | --- |
| `eqbench4_elo` | The policy's rating, `1200 + 400 / ln(10) * strength` |
| `eqbench4_elo_ci_low`, `eqbench4_elo_ci_high` | 95% interval from `rating_bootstrap_resamples` pair-stratified resamples |
| `eqbench4_rank`, `eqbench4_models_ranked` | The policy's place among the rated models |
| `eqbench4_ladder/<model>` | Each published model's rating in the same fit |
| `eqbench4_win_rate` | Mean comparison fraction |
| `eqbench4_trait/<name>` | Mean trait score, when the trait judge runs |
| `eqbench4_judged_conversations`, `eqbench4_persona_ended_rate`, `eqbench4_mean_policy_turns` | Statistics of the judged conversations |

`compute_repeat_metrics` reports each repeat's rating and win rate without an interval.

Gym passes only scored rows to `compute_metrics`, so `aggregate_metrics` counts the conversations left out of the
score over every row and adds:

| Metric | Meaning |
| --- | --- |
| `eqbench4_conversations` | All conversations, scored or left out |
| `eqbench4_policy_stopped_rate`, `eqbench4_policy_stopped_conversations` | Share and count the policy stopped (key metric) |
| `eqbench4_policy_stopped_rate/<cause>`, `eqbench4_policy_stopped/<cause>` | The same per cause: `context_limit`, `output_budget`, `content_filter`, `empty_reply` |
| `eqbench4_persona_failed_rate` | Share the persona could not continue (key metric) |

When any conversation was left out it logs a warning with the counts per cause and the usual fix.

## Configuration

`configs/eqbench4.yaml` pairs the server with `eqbench4_agent` and a `legacy_agent` environment server;
`configs/eqbench4_models.yaml` defines the persona and judge model servers. Key settings:
`judge_model_servers`, `judge_responses_create_params`, `judge_max_attempts`, `trait_judge_model_server`,
`judge_concurrency`, `judge_timeout_secs`, `reference_comparisons_path` and `rating_bootstrap_resamples`.

## Tests

```bash
gym env test --resources-server eqbench4
gym env test eqbench4
```

## Licensing

Code: Apache-2.0. Data: prepared from [EQ-bench/eqbench4](https://github.com/EQ-bench/eqbench4) under BSD-3-Clause
with the Commons Clause condition; nothing from it is committed. `data/example.jsonl` and the verifier fixture are
synthetic.

Dependencies
- nemo_gym: Apache 2.0
- numpy: BSD-3-Clause
