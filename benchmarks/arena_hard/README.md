# arena_hard

Gym implementation of the
[Arena Hard v0.1](https://github.com/lmarena/arena-hard-auto)
open-ended generation benchmark.

## What it tests

500 hard, open-ended user prompts. Each candidate rollout is judged
pairwise (both A↔B orderings) against a fixed **gpt-4-0314** baseline
via an LLM judge. See
[`resources_servers/arena_judge`](../../resources_servers/arena_judge/README.md)
for the judging protocol and metric details.

## Data

Runtime download only — benchmark JSONL is not committed. Run
[`prepare.py`](prepare.py) (or `gym eval prepare`) to populate
`data/arena_hard_benchmark.jsonl`. The prepare script fetches
questions and the baseline directly from the arena-hard-auto GitHub
repo, joins by `uid`, and emits one row per question with `question`,
`baseline_answer`, and `uid` at the top level. Arena-hard v0.1 has no
real sub-categories, so the upstream `category` field is dropped and
`arena_judge` falls through to its `default_category` (`hard_prompt`)
to pick the standard judge prompt.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark arena_hard

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark arena_hard

# Collecting rollouts
gym eval run --no-serve \
    --agent arena_hard_arena_judge_simple_agent \
    --input benchmarks/arena_hard/data/arena_hard_benchmark.jsonl \
    --output results/arena_hard_rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/prompts/generic/default.yaml
```

## Metrics

The headline number is the **Arena-Elo win-rate (%) vs baseline**,
computed by the `arena_judge` resources server as MLE logistic
regression over the pairwise battles with a 100-round bootstrap 95% CI.
Emitted keys:

- `arena_elo/score` — overall win-rate (0-100)
- `arena_elo/ci_lower` / `arena_elo/ci_upper` — bootstrap percentile CI bounds
- `arena_elo/invalid_scores` — count of judge calls that produced no
  parseable verdict

The server also emits pass@k / pass@1[avg-of-k] / majority@k for a
verdict-type decomposition (`wins`, `strict_wins`, `ties`, `losses`,
`double_wins`, `invalid_gen_base`), so a single run gives both the
Arena-Elo headline and a rollout-level verdict distribution without
extra post-processing.
