# EQ-Bench 4 Resources Server

Judges a policy's roleplay conversation against published EQ-Bench 4 conversations of the same scenario, and rates
the policy on the EQ-Bench 4 leaderboard scale. The [`eqbench4` benchmark](../../benchmarks/eqbench4/README.md)
describes the measurement contract, provenance and differences from upstream; this page covers the server.

## Verify

`eqbench4_agent` sends the row, the conversation (`transcript`), and how it ended (`conversation_status`):

| `conversation_status` | Result |
| --- | --- |
| `completed`, `persona_ended` | Judged against every opponent in the row; reward is the mean fraction |
| `policy_failed` | No judge calls; a loss to every opponent (fraction 0 each), reward 0, not masked |
| `persona_failed` | No judge calls; reward 0, masked as `eqbench4:persona_failed` (only from an older agent: the current agent fails the run instead, so it is retried) |

A conversation the policy stopped keeps its `policy_stop_reason` (`context_limit`, `output_budget`,
`content_filter`, or `empty_reply`, also when no reason is given). It is graded without a judge, inside
`observe_programmatic_grading`, so its grading record is complete and bound to the loss.

For each opponent the pairwise judge reads both conversations twice, policy first and opponent first, filled into the
row's `pairwise_prompt` with seeded respondent codes. Judge `n` of `judge_model_servers` reads comparison
`comparison_index` modulo the panel size. A reading's verdict is the JSON object in the reply (a fenced block, the
whole reply, or its outermost braces); a dimension counts when its value is a respondent's code followed by `+` signs.
Weighted margins from both readings are averaged; the policy's fraction is 0.5 plus or minus half the margin as a share
of the largest possible margin. A reading without a verdict (no JSON object, or one whose scored dimensions hold
neither `tie` nor a respondent's code) is asked again up to `judge_max_attempts` times in all; then the comparison
rests on the other order alone and records the failed one in `unreadable_orders`. When neither order has a verdict,
the row fails with a judge error and is masked as `judge_failed`.

When `trait_judge_model_server` is set, the trait judge scores the conversation's behavioural traits from the row's
`trait_prompt`. It runs inside the grading record, as an informational dependency of the one grading operation, and
never changes the reward; a failure is kept in `trait_judge_error`.

The response adds `comparisons` (per opponent: fraction, margins, and each judge reading with its verdicts and text),
`policy_turns`, `trait_scores`, `trait_judge_text`, `trait_judge_error`, and `evaluation_completed`: true for a scored
conversation, false for one left out of the score without judging.

## Metrics

`compute_metrics` reads the published comparisons at `reference_comparisons_path` and rates the policy together with
them by running EQ-Bench 4's own solver (`src/eqbench/ranking/soft_bt_solver.py` and its `config.py`), which the
benchmark's prepare script downloads, checks against pinned SHA-256 hashes and writes to `rating_solver_dir`.
`upstream_rating.py` checks the files again, imports them and calls `fit_soft_bt` and
`compute_local_delta_percentile_bootstrap_ratings`; Gym holds no rating code. Without the prepared files, or when some
comparisons are against a model the published graph does not hold, no rating is reported and a warning says why; the
other metrics are:

| Metric | Meaning |
| --- | --- |
| `eqbench4_elo` | The policy's rating, `1200 + 400 / ln(10) * strength` |
| `eqbench4_elo_ci_low`, `eqbench4_elo_ci_high` | 95% interval from `rating_bootstrap_resamples` pair-stratified resamples |
| `eqbench4_rank`, `eqbench4_models_ranked` | The policy's place among the rated models |
| `eqbench4_ladder/<model>` | Each published model's rating in the same fit |
| `eqbench4_win_rate` | Mean comparison fraction |
| `eqbench4_unrated_comparisons` | Comparisons against models missing from the published comparisons (such as the synthetic example rows' opponent); when nonzero, no rating is reported, since the fit would leave them out |
| `eqbench4_trait/<name>` | Mean trait score, when the trait judge runs |
| `eqbench4_judged_conversations`, `eqbench4_persona_ended_rate`, `eqbench4_mean_policy_turns` | Statistics of the judged conversations; a conversation the policy stopped was not judged and is not counted here, though its losses count in the win rate and rating |

`compute_repeat_metrics` reports each repeat's rating and win rate without an interval.

`aggregate_metrics` counts, over every row, the conversations the policy stopped and those left out of the score, and
adds:

| Metric | Meaning |
| --- | --- |
| `eqbench4_conversations` | All conversations |
| `eqbench4_policy_stopped_rate`, `eqbench4_policy_stopped_conversations` | Share and count the policy stopped, each scored as a loss (key metric) |
| `eqbench4_policy_stopped_rate/<cause>`, `eqbench4_policy_stopped/<cause>` | The same per cause: `context_limit`, `output_budget`, `content_filter`, `empty_reply` |
| `eqbench4_persona_failed_rate` | Share an older agent reported the persona could not continue (key metric) |

When the policy stopped any conversation it logs a warning with the counts per cause and the usual fix.

## Configuration

`configs/eqbench4.yaml` pairs the server with `eqbench4_agent` and a `legacy_agent` environment server, and admits only
that agent (`allowed_agents`): the conversation with the simulated persona is the agent's, so EQ-Bench 4 runs only with
`eqbench4_agent`. `configs/eqbench4_models.yaml` defines the persona and judge model servers. Key settings:
`judge_model_servers`, `judge_responses_create_params`, `judge_max_attempts`, `trait_judge_model_server`,
`judge_concurrency`, `judge_timeout_secs`, `reference_comparisons_path`, `rating_solver_dir` and
`rating_bootstrap_resamples`.

## Tests

```bash
gym env test --resources-server eqbench4
gym env test eqbench4
```

## Licensing

Code: Apache-2.0. Data and the rating solver: downloaded at preparation from
[EQ-bench/eqbench4](https://github.com/EQ-bench/eqbench4) under BSD-3-Clause with the Commons Clause condition;
nothing from it is committed. `data/example.jsonl`, the verifier fixture and the tests' stand-in solver are synthetic.

Dependencies
- nemo_gym: Apache 2.0
