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
- The vLLM adapter records zero usage for its synthetic empty response when the
  backend rejects a request for exceeding context before generation. This keeps
  earlier sub-step token totals available; it does not replace unknown usage on
  actual generated responses. Older saved records are not repaired automatically.
- If any generated response lacks usage, suppress all aggregate token statistics,
  omit the token means, and report `token_usage_complete: false`.
  Accuracy, coverage, and raw rollout records remain available. Missing optional
  reasoning/cache breakdowns do not invalidate known input/output/total usage.

`step_usage` records each step's number, status, and usage. Coverage counts are
`num_subproblems`, `num_generated_steps`, and `num_steps_with_usage`.

Generated entries also retain `response_id`, `response_status`,
`incomplete_details`, and `error` from that step's response, independently of
model-call capture. Missing stop metadata stays null; it does not imply a
completed generation. Use these fields to detect limits in any generated step:
the top-level response status describes only the final generation. Prefilled,
context-rejected, and skipped steps have no response metadata; their step
`status` distinguishes them from generated responses. Older rollouts may lack
these fields and cannot establish a zero truncation rate for multi-step problems.

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


The vLLM adapter marks locally rejected context overflows in Responses metadata
as `nemo_gym_context_window_exceeded: "true"`. The agent records those steps as
`context_window_exceeded` and skips remaining steps, even with overflow propagation
disabled. An ordinary zero-token `max_output_tokens` response has no such marker
and remains a generated step; zero usage alone does not identify a context error.
Rejected requests count as zero tokens processed (including cached/reasoning
details), rather than the rejected prompt's length. All vLLM-backed benchmarks
that encounter these requests can therefore report slightly lower generic token
means, since the requests previously had unknown usage and were excluded.
