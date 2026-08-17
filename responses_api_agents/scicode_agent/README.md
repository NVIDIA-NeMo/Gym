# SciCode Agent

Custom multi-step agent for the SciCode benchmark. For each problem it loops over the sub-steps,
generating Python code one sub-step at a time and accumulating it (each sub-step's prompt includes
the model's own code from previous sub-steps), then submits the accumulated per-step solutions to
the SciCode resources server's `/verify` for execution.

It also reports the headline `subtask_accuracy` metric (total sub-steps passed / total, over all
rollouts) via `compute_metrics` / `get_key_metrics` — these live on the agent because
`/aggregate_metrics` runs on the agent server.

## Configuration

- `resources_server`: the SciCode resources server instance to verify against
- `model_server`: the model server used for generation
- `prompt_fpath`: per-sub-step prompt template the agent fills each step
  (e.g. `benchmarks/scicode/prompts/default.yaml`)
- `with_background` (default `true`): inject each sub-step's scientific background into the prompt
  context

The full wiring (resources server + this agent + dataset) lives in `benchmarks/scicode/config.yaml`.
