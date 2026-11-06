# indic-arena-hard-v2

Gym implementation of
[AI4Bharat Indic Arena Hard v2](https://huggingface.co/datasets/ai4bharat/indic-arena-hard-v2),
a translated open-ended generation benchmark.

## What it tests

**504 prompts per language**, translated into 14 Indian languages, across
two categories:

- `hard_prompt` — 363 reasoning-heavy technical and analytical queries
- `creative_writing` — 141 open-ended creative tasks

Each candidate rollout is judged **pointwise by default**, without a
baseline, on correctness, relevance, completeness, and language quality.
Optional pairwise evaluation compares answers against matching-language
baselines using both answer orderings.

See [`resources_servers/indic_arena_judge`](../../../resources_servers/indic_arena_judge/README.md)
for the judging protocol and metrics. The Indic judge is separate from
the unchanged English Arena Hard benchmark and judge.

## Data

The [dataset card](https://huggingface.co/datasets/ai4bharat/indic-arena-hard-v2/blob/aebdc931858cb6df854f16a0513e184fc38a3b1e/README.md)
identifies **504 originally English prompts** among Arena Hard v2's 750
source prompts. AI4Bharat translates those 504; the other 246 prompts,
originally written in other languages, are excluded.

The source has 504 rows with translation columns. Preparation includes all
14 translations and the English control by default: **7,056 translated
tasks**, or **7,560 tasks including English**. Supported language codes:
`en`, `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`,
`te`, `ur`.

Runtime download only — generated benchmark JSONL is not committed. Run
[`prepare.py`](prepare.py) or `gym eval prepare` to populate
`data/indic_arena_hard_v2_benchmark.jsonl`. Hub preparation pins revision
`aebdc931858cb6df854f16a0513e184fc38a3b1e`.

Prepared rows preserve source categories and IDs, with `question`,
`baseline_answer`, `category`, `uid`, and `language` at the top level.
The source `uid` becomes `row_id`; task IDs use `<language>:<row_id>`.
The dataset supplies no baseline answers; pointwise evaluation needs none.

## Example usage

Run from the repository root. Configure the candidate model through Gym's
usual model settings and set the
[Indic judge endpoint variables](../../../resources_servers/indic_arena_judge/README.md#configuring-the-judge).

```bash
# Prepare benchmark data
gym eval prepare --benchmark indic/indic-arena-hard-v2

# Running servers (pointwise is the default)
gym env start \
    --model-type vllm_model \
    --benchmark indic/indic-arena-hard-v2

# Collecting rollouts
gym eval run --no-serve \
    --agent indic_arena_hard_v2_simple_agent \
    --input benchmarks/indic/indic-arena-hard-v2/data/indic_arena_hard_v2_benchmark.jsonl \
    --output results/indic_arena_hard_v2_rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/prompts/generic/default.yaml
```

See [Additional usage](usage.md) for local data preparation, language
selection, pairwise baselines, and a three-task smoke run.

## Metrics

The default headline number is **`pointwise/score` (0–100)**: the equally
weighted mean of four rubric dimensions, each rated 0–4, rescaled to 100.
Repeats are averaged within each task before averaging tasks. Emitted keys:

- `pointwise/score` — overall rubric score
- `pointwise/{correctness,relevance,completeness,language_quality}` —
  dimension means on the 0–4 scale
- `pointwise/language/<code>/score` — per-language rubric score
- `pointwise/invalid_scores` — malformed judgments excluded from score means

Pairwise mode reports `arena_elo/score`, confidence bounds, category
breakdowns, and invalid verdict counts through the existing Arena judge.
Evaluate each language separately for per-language pairwise scores.

Pointwise rubric scores are not Arena Hard win rates. Pairwise results
also depend on the translated prompts and chosen baselines, so they are
not directly comparable to the original English benchmark.
