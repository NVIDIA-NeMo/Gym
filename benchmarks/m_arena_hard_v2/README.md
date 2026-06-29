# m_arena_hard_v2

Gym implementation of the multilingual
[m-ArenaHard-v2.0](https://huggingface.co/datasets/CohereLabs/m-ArenaHard-v2.0)
open-ended generation benchmark — the m-Arena translation of
[Arena Hard v2](https://github.com/lmarena/arena-hard-auto) covering
23 languages.

## What it tests

~11,454 hard, open-ended user prompts (498 per language × 23 languages)
across two categories preserved from upstream:

- `hard_prompt` — reasoning-heavy technical and analytical queries
- `creative_writing` — open-ended creative tasks

Each candidate rollout is judged pairwise (both A↔B orderings) against
its baseline via an LLM judge. See
[`resources_servers/arena_judge`](../../resources_servers/arena_judge/README.md)
for the judging protocol and metric details.

## Data

Runtime download only — benchmark JSONL is not committed. Run
[`prepare.py`](prepare.py) (or `gym eval prepare`) to populate
`data/m_arena_hard_v2_benchmark.jsonl`. The prepare script loads the HF
dataset across all 23 language configs, iterates each split, and emits
one row per `(language, question_id)` with `uid`, `question`,
`baseline_answer`, `language`, `category` (per-row), and
`subset_for_metrics` (the language code) at the top level.

The HF dataset requires accepting Cohere's terms — set `HF_TOKEN`
before running prepare if your environment doesn't have it cached.

### `--baseline-file` (required for end-to-end judging)

Upstream m-ArenaHard-v2.0 ships **no baselines**. To produce a
non-empty `baseline_answer`, supply `--baseline-file` pointing at a
JSONL with rows `{language, question_id, generation}` (the natural
output shape of a Skills/Gym generation run); the prepare script joins
by `(language, question_id)`. Without it, `baseline_answer` is the
empty string and the file is only useful for prompt inspection.

```bash
# All 23 languages, no baseline (prompt inspection only)
python benchmarks/m_arena_hard_v2/prepare.py

# Subset of languages with a pre-generated baseline
python benchmarks/m_arena_hard_v2/prepare.py \
    --languages en es \
    --baseline-file /path/to/baseline_generations.jsonl
```

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark m_arena_hard_v2

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark m_arena_hard_v2

# Collecting rollouts
gym eval run --no-serve \
    --agent m_arena_hard_v2_arena_judge_simple_agent \
    --input benchmarks/m_arena_hard_v2/data/m_arena_hard_v2_benchmark.jsonl \
    --output results/m_arena_hard_v2_rollouts.jsonl \
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

## Generation sanitization

The benchmark sets `sanitize_generations: true` on its
`arena_judge` resources server (see [`config.yaml`](config.yaml)) to
scrub UTF-8 surrogate halves and NULs from candidate and baseline
generations before judging — mirrors Skills'
`++sanitize_generations=true` for the multilingual variant.

## Deferred follow-ups

- **Per-language Arena-Elo aggregation.** Rows already carry
  `language` and `subset_for_metrics: <language_code>`; wiring those
  into `arena_judge`'s subset-aware metric output is a follow-up. For
  now the headline is global + per-category, not per-language.
