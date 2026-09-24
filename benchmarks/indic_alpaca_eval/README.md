# Indic AlpacaEval

Integrates the local or pinned Hugging Face dataset
[anushakamathofficial/indic_alpaca_eval](https://huggingface.co/datasets/anushakamathofficial/indic_alpaca_eval).
It reuses the same generic user prompt, simple agent, and reference-free rubric
as the Indic Arena Hard integration. No existing English AlpacaEval integration
was present in this checkout.

## How AlpacaEval is normally evaluated

The [upstream AlpacaEval package](https://github.com/tatsu-lab/alpaca_eval) compares
a candidate answer with a reference answer for each of 805 instructions. Its
AlpacaEval 2 default uses GPT-4 Turbo reference outputs and the
`weighted_alpaca_eval_gpt4_turbo` annotator. It reports raw preference win rate
and length-controlled win rate. [Length control](https://arxiv.org/abs/2404.04475)
estimates preference after adjusting for answer-length differences; it does not
truncate answers to equal lengths.

Translated prompts need reference answers for those exact translated instructions.
An English reference must not be joined to an Indic prompt by row ID alone.
Changing the prompts, reference model, or judge creates a different evaluation;
the resulting scores are not directly comparable to the official English leaderboard.

## Scoring in Gym

The default config uses the shared `arena_judge` in **pointwise** mode, because
the supplied dataset contains instructions and translation metadata but no answers.
This custom rubric rates correctness, relevance, completeness, and language quality
from 0 to 4. Their equally weighted mean is rescaled to `pointwise/score` (0–100);
reward is that score divided by 100. Per-language metrics are also reported.
This is neither AlpacaEval win rate nor its length-controlled metric.

Empty answers score zero. Invalid judge JSON is counted and excluded from rubric
means. Judge transport failures go to the failures sidecar. Inspect coverage,
`pointwise/invalid_scores`, and failures alongside scores. Separate Responses API
reasoning items are excluded from judging and export; configure the candidate
adapter's reasoning parser for thinking models.

For standard pairwise evaluation, export the answers and use the upstream package
as described below. Do not select the Arena Hard pairwise prompt: it implements
a different protocol and reports Arena-Elo.

## Data and preparation

Revision: `5a173582c9e9f184e6a8fdefb0fb12ae485d56fa`.
The local source has 805 rows in each of 15 languages:
`as bn en gu hi kn ml mr ne or pa sa ta te ur` (12,075 rows).
All 3,402 translated rows pending human review are retained. The original
`instruction`, `row_id`, `row_index`, and review fields are preserved.
`question` holds the same instruction for the generic Gym prompt;
`uid` is `<language>:<row_id>`, `language` and `subset_for_metrics` hold
the language code, and `language_name` holds the original full language name.

The source dataset is labeled **CC-BY-NC-4.0**. Integration code is Apache-2.0;
dataset files are read/downloaded at runtime and are not committed.
Local preparation uses files as supplied without enforcing the Hub revision pin.

Run from the repository root:

```bash
DATASET_DIR=/path/to/huggingface_datasets/anushakamathofficial/indic_alpaca_eval
python -m benchmarks.indic_alpaca_eval.prepare --dataset-dir "$DATASET_DIR"

# A subset, entirely offline:
python -m benchmarks.indic_alpaca_eval.prepare \
    --dataset-dir "$DATASET_DIR" --languages hi ta

# Or use the pinned Hub dataset:
gym eval prepare --benchmark indic_alpaca_eval
```

Preparation replaces the generated JSONL after validating all selected rows.
Optional references use the same format as the Arena baseline files:

```json
{"language": "hi", "row_id": "helpful_base-0000", "generation": "<reference answer>"}
```

Pass `--reference-file path/to/references.jsonl` during preparation.
Missing, duplicate, or empty references fail before replacing the prepared data.
The custom pointwise rubric ignores these optional references; they are retained
for export to the pairwise evaluator.

## Gemini smoke test

The [Gemini preset](gemini_pointwise.yaml) uses
`gcp/google/gemini-3.8-flash` at `https://inference-api.nvidia.com/v1` as
candidate and judge, with temperature 0 and a 4,096-token output limit.
Same-model judging is a self-evaluation smoke test.

Validated locally with one English and two Hindi instructions: all three candidate
responses completed without truncation, all three judge results were valid, and
the rollout health checks passed. Both language exports were also checked.
This verifies the integration, not model quality on the complete benchmark.

```bash
read -rsp 'NVIDIA API key: ' NVIDIA_API_KEY
export NVIDIA_API_KEY
gym env start --config benchmarks/indic_alpaca_eval/gemini_pointwise.yaml
```

In a second terminal, from the same repository with Gym activated:

```bash
gym eval run --no-serve \
    --agent indic_alpaca_eval_simple_agent \
    --input benchmarks/indic_alpaca_eval/data/indic_alpaca_eval_benchmark.jsonl \
    --output results/indic_alpaca_eval_smoke.jsonl \
    --prompt-config benchmarks/prompts/generic/default.yaml \
    --limit 3 --num-repeats 1 --concurrency 1 \
    --max-output-tokens 4096 --temperature 0
```

Use Ctrl+C in the server terminal after evaluation. For another candidate, load
`config.yaml` with its model configuration and set `ARENA_JUDGE_BASE_URL`,
`ARENA_JUDGE_API_KEY`, and `ARENA_JUDGE_MODEL` for the judge. For generation
without Gym judging, start servers with `++skip_verification=true`; no rubric
score should then be interpreted as an evaluation result.

## Export for upstream pairwise evaluation

After collecting one response per prepared instruction:

```bash
python -m benchmarks.indic_alpaca_eval.export \
    --rollouts-file results/indic_alpaca_eval_rollouts.jsonl \
    --dataset-file benchmarks/indic_alpaca_eval/data/indic_alpaca_eval_benchmark.jsonl \
    --language hi --generator candidate-model \
    --output-file results/alpaca_hi/model_outputs.json \
    --reference-output-file results/alpaca_hi/reference_outputs.json \
    --reference-generator reference-model
```

The reference options require preparation with `--reference-file`. Omit them
to export candidates alone. The exporter requires complete selected-language
coverage, rejects duplicate repeats and mismatched instructions, and keeps empty
or truncated candidate answers rather than silently dropping them.

In an environment with the upstream `alpaca-eval` package and its judge credentials,
evaluate translated pairs with explicit matching-language references:

```bash
alpaca_eval \
    --model_outputs results/alpaca_hi/model_outputs.json \
    --reference_outputs results/alpaca_hi/reference_outputs.json \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --fn_metric get_winrate --sort_by win_rate
```

This reports raw pairwise win rate for a custom translated benchmark. The English
length-control calibration has not been validated for these translations. For the
original English protocol, export `en` and use the upstream defaults (including its
reference outputs and length-controlled metric). All 805 local English instructions
were checked to match the official reference set exactly at upstream dataset revision
`2edc6fad8be6b14ea7230aabfd08188da6b8b814`. The current integration has not reproduced
official leaderboard scores and remains experimental with `verified: false`.
