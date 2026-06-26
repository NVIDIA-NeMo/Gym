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
[`prepare.py`](prepare.py) (or `gym eval prepare`) to populate
`data/arena_hard_v2_benchmark.jsonl`. The prepare script fetches
questions and both baselines directly from the arena-hard-auto GitHub
repo, joins by `uid`, and emits one row per question with `question`,
`baseline_answer`, `category`, and `uid` at the top level.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark arena_hard_v2

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark arena_hard_v2

# Collecting rollouts
gym eval run --no-serve \
    --agent arena_hard_v2_arena_judge_simple_agent \
    --input benchmarks/arena_hard_v2/data/arena_hard_v2_benchmark.jsonl \
    --output results/arena_hard_v2_rollouts.jsonl \
    --num-repeats 4
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
