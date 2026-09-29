# SciCode Agent

Custom multi-step agent for the SciCode benchmark. For each problem it loops over the sub-steps,
generating Python code one sub-step at a time and accumulating it (each sub-step's prompt includes
the model's own code from previous sub-steps), then submits the accumulated per-step solutions to
the SciCode resources server's `/verify` for execution.

It also reports the headline `subtask_accuracy` metric (total sub-steps passed / total, over all
rollouts) via `compute_metrics` / `get_key_metrics` — these live on the agent because
`/aggregate_metrics` runs on the agent server.

## Token statistics

`response.usage` sums usage across the problem's sub-steps; response text remains
the final generation. Output includes reported reasoning tokens, and input includes
prior-step code sent again as context.

For each of `input`, `output`, and `total`, `key_metrics` exposes:

| Metric | Definition |
| --- | --- |
| `mean/<type>_tokens_per_problem` | Token sum / problem attempts; also `mean/<type>_tokens` |
| `mean/<type>_tokens_per_subproblem` | Token sum / non-prefilled benchmark sub-steps across all attempts |

Repeats count separately. The subproblem denominator includes final, rejected, and
skipped steps; only prefilled steps are excluded. `generation_coverage` is generated
steps / non-prefilled steps. With no eligible sub-steps, the subproblem mean is omitted.

- Count all reported usage, including incorrect or invalid code and generations
  that hit a token limit. Incorrect or invalid code does not stop later steps.
- Pre-generation context rejections, prefilled steps, and skipped steps use zero
  tokens. Earlier generations still count. Aborted rollouts are excluded from
  completed-rollout metrics.
- If any generated response lacks usage, suppress all aggregate token statistics,
  omit the token means, and report `token_usage_complete: false`.
  Accuracy, coverage, and raw rollout records remain available. Missing optional
  reasoning/cache breakdowns do not invalidate known input/output/total usage.

`step_usage` records each step's number, status, and usage. Coverage counts are
`num_subproblems`, `num_generated_steps`, and `num_steps_with_usage`.

Rollouts and aggregates use `token_usage_version: 1`. Legacy records retain
final-step metrics. Mixing versions is rejected.

## Configuration

- `resources_server`: the SciCode resources server instance to verify against
- `model_server`: the model server used for generation
- `prompt_fpath`: per-sub-step prompt template the agent fills each step
  (e.g. `benchmarks/scicode/prompts/default.yaml`)
- `with_background` (default `true`): inject each sub-step's scientific background into the prompt
  context

The full wiring (resources server + this agent + dataset) lives in `benchmarks/scicode/config.yaml`.
