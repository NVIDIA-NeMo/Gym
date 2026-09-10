# SciCode Agent

Custom multi-step agent for the SciCode benchmark. For each problem it loops over the sub-steps,
generating Python code one sub-step at a time and accumulating it (each sub-step's prompt includes
the model's own code from previous sub-steps), then submits the accumulated per-step solutions to
the SciCode resources server's `/verify` for execution.

It also reports the headline `subtask_accuracy` metric (total sub-steps passed / total, over all
rollouts) via `compute_metrics` / `get_key_metrics` — these live on the agent because
`/aggregate_metrics` runs on the agent server.

## Token statistics

Each rollout records `response.usage` summed across all generated sub-steps in that
problem attempt. Output tokens include the provider's reported reasoning tokens;
input tokens include prior-step code sent again as context. The response text remains
the final step's generation.

The agent exposes six headline metrics in `key_metrics`:

| Metric | Definition |
| --- | --- |
| `mean/input_tokens_per_problem` | Total input tokens / problem attempts (also `mean/input_tokens`) |
| `mean/output_tokens_per_problem` | Total output tokens / problem attempts (also `mean/output_tokens`) |
| `mean/total_tokens_per_problem` | Total input + output tokens / problem attempts (also `mean/total_tokens`) |
| `mean/input_tokens_per_subproblem` | Total input tokens / benchmark sub-step count across attempts, excluding prefilled steps |
| `mean/output_tokens_per_subproblem` | Total output tokens / benchmark sub-step count across attempts, excluding prefilled steps |
| `mean/total_tokens_per_subproblem` | Total input + output tokens / benchmark sub-step count across attempts, excluding prefilled steps |

Repeats count as separate attempts. Sub-step usage is pooled across problems, so
problems with more steps contribute more observations to the subproblem mean.
The subproblem denominator is fixed by the benchmark, including the final step,
context-rejected steps, and skipped remaining steps. Only prefilled reference-code
steps are excluded. A problem that generates no steps uses zero tokens; a
collection with no non-prefilled subproblems has a null subproblem mean.
`generation_coverage` reports generated steps / non-prefilled benchmark steps.
Report it alongside accuracy and token means: early stopping reduces cost but
also reduces coverage. For three subproblems, generating only one 1,000-token
response gives 1,000 output tokens per problem, 333.33 per subproblem, and 1/3
generation coverage.

Generations that reach an output or context limit **are counted**, including all
reported tokens and the attempt in the subproblem denominator. A pre-generation
context-window rejection records zero input, output, and total tokens, as do
prefilled and skipped steps. Prior generations' usage is still counted.
If a model adapter turns a rejection into an empty response without usage, that
response is treated as having unknown usage, rather than assuming zero tokens.

Correctness does not filter token usage. The agent generates all sub-steps before
verification, using its prior generated code even if that code is incorrect.
For example, two models that each generate three 1,000-output-token sub-steps
both use 3,000 output tokens per problem and 1,000 per subproblem, whether they
solve one sub-step correctly or all three. These metrics measure tokens per attempted solution,
not tokens per correct solution.

Code extraction does not validate Python syntax: it takes the first code block,
or the raw response if there is no code fence. Empty or invalid code still counts
as a generated attempt, with all reported tokens, and does not stop later steps.
An unexpected exception that aborts the rollout is a run failure; its partial
usage is not represented in completed-rollout metrics.

Compact `step_usage` records retain each `step_number`, its status (`generated`,
`prefilled`, `context_window_exceeded`, or `skipped`), and its provider usage. A
generated step with missing usage has `usage: null` and makes its problem's total
usage unknown. If even one generated response lacks usage, the aggregate endpoint
suppresses all generic token statistics (including per-task and per-repeat
statistics), sets the six named token means to null, and reports
`token_usage_complete: false`. It preserves raw rollout records for diagnosis;
accuracy and coverage remain available. `num_subproblems`, `num_generated_steps`,
and `num_steps_with_usage` expose the counts. Missing reasoning/cache breakdowns
alone do not invalidate known input/output/total usage.

New rollouts and aggregates carry `token_usage_version: scicode-v1`. Older rollouts
have no version and their `mean/output_tokens` covers only the final step; they
cannot be backfilled from rollout files alone. Aggregating legacy-only rollouts
preserves their existing metrics without adding the six new metrics. Mixing old
and new accounting versions raises an error to prevent misleading comparisons.

## Configuration

- `resources_server`: the SciCode resources server instance to verify against
- `model_server`: the model server used for generation
- `prompt_fpath`: per-sub-step prompt template the agent fills each step
  (e.g. `benchmarks/scicode/prompts/default.yaml`)
- `with_background` (default `true`): inject each sub-step's scientific background into the prompt
  context

The full wiring (resources server + this agent + dataset) lives in `benchmarks/scicode/config.yaml`.
