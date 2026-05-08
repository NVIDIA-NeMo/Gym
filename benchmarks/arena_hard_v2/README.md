# arena_hard_v2

Gym implementation of the
[Arena Hard v2](https://github.com/lmarena/arena-hard-auto) open-ended
generation benchmark.

## What it tests

~750 hard, open-ended user prompts across two categories:

- `hard_prompt` — reasoning-heavy technical and analytical queries,
  judged against an **o3-mini** baseline
- `creative_writing` — open-ended creative tasks, judged against a
  **gemini-2.0-flash-001** baseline

Each candidate rollout is judged pairwise (both A↔B orderings) against
its category-specific baseline via an LLM judge. See
[`resources_servers/arena_judge`](../../resources_servers/arena_judge/README.md)
for the judging protocol and metric details.

## Data

Runtime download only — benchmark JSONL is not committed. Run
[`prepare.py`](prepare.py) (or `ng_prepare_benchmark`) to populate
`data/arena_hard_v2_benchmark.jsonl`. The prepare script fetches
questions and both baselines directly from the arena-hard-auto GitHub
repo, joins by `uid`, and emits one row per question with `question`,
`baseline_answer`, `category`, and `uid` at the top level.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/arena_hard_v2/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/arena_hard_v2/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=arena_hard_v2_arena_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/arena_hard_v2/data/arena_hard_v2_benchmark.jsonl \
    +output_jsonl_fpath=results/arena_hard_v2_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

The headline number is the **Arena-Elo win-rate (%) vs baseline**,
computed by the `arena_judge` resources server as MLE logistic
regression over the pairwise battles with a 100-round bootstrap 95% CI.
Emitted keys:

- `arena_elo/score` — overall win-rate (0-100)
- `arena_elo/ci_lower` / `arena_elo/ci_upper` — bootstrap percentile CI bounds
- `arena_elo/{hard_prompt,creative_writing}/score` + CIs — per-category
  breakdown
- `arena_elo/invalid_scores` — count of judge calls that produced no
  parseable verdict

The server also emits pass@k / pass@1[avg-of-k] / majority@k for a
verdict-type decomposition (`wins`, `strict_wins`, `ties`, `losses`,
`double_wins`, `invalid_gen_base`), so a single run gives both the
Arena-Elo headline and a rollout-level verdict distribution without
extra post-processing.
