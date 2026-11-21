# indic_arena_judge

LLM-judge resources server for
[Indic Arena Hard v2](../../benchmarks/indic/indic-arena-hard-v2/README.md),
with **pointwise scoring by default** and optional pairwise evaluation.
It stays separate from the English `arena_judge` and reuses its pairwise
implementation through inheritance.

## What it does

### Pointwise (default)

For each nonempty candidate answer, the server makes **one** judge call
using the [pointwise rubric](prompts/pointwise.yaml). The judge rates
correctness, relevance, completeness, and language quality from 0 to 4.
No baseline is required. The resource server returns:

- `pointwise_score` — the four scores averaged and rescaled to 0–100
- `reward` — `pointwise_score / 100`
- `pointwise_scores` and `judgement_pointwise` — parsed scores and raw
  judge output, retained in the rollout

Empty answers score zero without a judge call. Malformed judgments receive
reward zero and `invalid_pointwise: true`; they are counted and excluded
from rubric means. Judge API failures use Gym's failures sidecar.

### Pairwise (opt-in)

A nonempty matching-language `baseline_answer` is required. The server
then delegates to the [English Arena judge](../arena_judge/README.md):

1. `gen-base` — A = candidate, B = baseline
2. `base-gen` — A = baseline, B = candidate

The existing category prompts, verdict parsing, and Arena-Elo aggregation
are reused unchanged. Missing or blank baselines are rejected before
calling the judge.

## Data schema

Each JSONL row carries these top-level fields, described in
[`task_data.py`](task_data.py):

- `question` — prompt sent to the candidate model
- `baseline_answer` — required for pairwise; unused by pointwise
- `category` — `hard_prompt` or `creative_writing`, selecting the pairwise prompt
- `uid` — task ID, including the language code for prepared benchmark rows
- `language` — language code used for pointwise per-language metrics

The five committed examples in `data/example.jsonl` are synthetic.
Benchmark data is prepared separately at runtime.

## Example usage

Configure the judge endpoint below and the candidate model through Gym's
usual model settings. Run from the repository root:

```bash
# Running servers (pointwise is the default)
gym env start \
    --model-type vllm_model \
    --resources-server indic_arena_judge

# Collecting rollouts (5-example smoke test)
gym eval run --no-serve \
    --agent indic_arena_judge_simple_agent \
    --input resources_servers/indic_arena_judge/data/example.jsonl \
    --output results/indic_arena_judge_rollouts.jsonl \
    --num-repeats 1
```

For standalone pairwise runs, set `evaluation_mode: pairwise` in
[`configs/indic_arena_judge.yaml`](configs/indic_arena_judge.yaml).
For the benchmark, pass `++indic_arena_hard_v2_judge_mode=pairwise` when
starting servers; see [pairwise evaluation](../../benchmarks/indic/indic-arena-hard-v2/usage.md#pairwise-evaluation).

## Configuring the judge

Use an OpenAI-compatible endpoint that supports `/v1/chat/completions`:

- `INDIC_ARENA_JUDGE_BASE_URL` — endpoint base URL
- `INDIC_ARENA_JUDGE_API_KEY` — endpoint API key
- `INDIC_ARENA_JUDGE_MODEL` — model identifier accepted by the endpoint

Set these before starting servers. No judge model is selected implicitly.
The default judge settings are **temperature 0** and **4,096 maximum
output tokens**. Indic instances and variables are separate from the
English benchmark's `ARENA_JUDGE_*` configuration.

## Metrics

Pointwise metrics average valid repeats within each task, then average
tasks. `compute_metrics()` emits:

- **`pointwise/score`** — overall rubric score (0–100)
- **`pointwise/{correctness,relevance,completeness,language_quality}`** —
  dimension means (0–4)
- **`pointwise/language/<code>/score`** — per-language score, alongside
  per-language dimensions and counts
- **`pointwise/valid_scores`**, **`pointwise/invalid_scores`**, and
  **`pointwise/scored_tasks`** — judgment and task counts

Check invalid counts alongside the score. These are custom rubric scores,
not official Arena Hard win rates.

Pairwise mode emits `arena_elo/score`, bootstrap confidence bounds,
category breakdowns, invalid verdict counts, and win/tie/loss metrics.
See the [Arena judge metrics](../arena_judge/README.md#metrics) for details.
Pairwise aggregation is by category; run each language separately for
per-language scores.
