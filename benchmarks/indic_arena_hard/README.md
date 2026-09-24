# Indic Arena Hard

Integrates [anushakamathofficial/indic_arena_hard](https://huggingface.co/datasets/anushakamathofficial/indic_arena_hard)
using the same `arena_judge`, `simple_agent`, `policy_model`, and generic prompt
as the [English Arena Hard benchmark](../arena_hard/README.md). Judge prompts,
sampling settings (temperature 0, maximum 4096 tokens), side-swapped comparisons,
and scoring come from the shared English configuration.

## Data

Hub preparation pins dataset revision `cc08e98842cd04d659092dc8b13cf00a848b435c`.
By default it includes all 15 configurations: `en`, `as`, `bn`, `gu`, `hi`, `kn`,
`ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`, `te`, and `ur`. There are 504 English
rows and 7,055 translated rows (503 Urdu rows, 504 in each other language).
No missing translations are synthesized and no review-status filtering is applied.

`prompt` becomes `question`; `uid` is `<language_code>:<row_id>` so translations
remain distinct tasks. The original `row_id`, `row_index`, `language_code`, and
all translation-review fields are preserved. The full language name is retained
as `language_name`; `language` and `subset_for_metrics` hold the language code.
Every task uses the standard `hard_prompt` judge category.

Data is read from a local download or downloaded at runtime; generated JSONL is
ignored by Git. Pass `--dataset-dir` to the downloaded dataset repository containing
`data/<language>/test.parquet` to prepare entirely offline, without a token.
Local mode uses the files as supplied and discovers the languages present on disk;
it does not enforce the Hub revision pin. For Hub downloads, use an existing
Hugging Face login (`hf auth login`) or set `HF_HOME` to your Hugging Face cache
directory. Tokens belong in the private local credential store, never in configs.

## Baseline answers

This dataset contains **no baseline answers**. Its 504-row English split differs
from the upstream 500-question Arena Hard v0.1 evaluation set.
Supply answers generated for these exact prompts in each
language; the prepare script does not substitute English answers for translations.

The baseline JSONL format is:

```json
{"language": "hi", "row_id": "<source row_id>", "generation": "<baseline answer>"}
```

Rows are joined by `(language, row_id)`. Missing, duplicate, and empty baseline
answers fail preparation before replacing the output. Without `--baseline-file`,
preparation warns and emits empty `baseline_answer` values, matching the existing
multilingual Arena Hard preparation workflow. That output is only for prompt
inspection or generation; do not use it to report pairwise benchmark scores.

## Usage

Run from the repository root with the same Gym environment as the English benchmark.

```bash
# Prepare an existing local download (no network or token needed).
DATASET_DIR=/path/to/huggingface_datasets/anushakamathofficial/indic_arena_hard
python benchmarks/indic_arena_hard/prepare.py --dataset-dir "$DATASET_DIR"

# The same local source can be passed through Gym.
gym eval prepare --benchmark indic_arena_hard \
    +prepare_script_args.dataset_dir="$DATASET_DIR"

# Download all prompts for inspection (no baseline answers).
gym eval prepare --benchmark indic_arena_hard

# Prepare selected languages with baseline answers before judging.
python benchmarks/indic_arena_hard/prepare.py \
    --dataset-dir "$DATASET_DIR" \
    --languages hi ta \
    --baseline-file path/to/baselines.jsonl

# Use the same ARENA_JUDGE_BASE_URL, ARENA_JUDGE_API_KEY, ARENA_JUDGE_MODEL,
# and policy-model configuration as the English benchmark.
gym env start --model-type vllm_model --benchmark indic_arena_hard

gym eval run --no-serve \
    --agent indic_arena_hard_arena_judge_simple_agent \
    --input benchmarks/indic_arena_hard/data/indic_arena_hard_benchmark.jsonl \
    --output results/indic_arena_hard_rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/prompts/generic/default.yaml
```

Omit `--languages` to prepare every configuration, including English, with a
complete baseline file. Use `--languages en` for the dataset's English control.
Re-running preparation without baselines replaces the output with prompts only.

## Metrics and validation status

The shared [Arena judge](../../resources_servers/arena_judge/README.md) reports
`arena_elo/score`, bootstrap confidence bounds, invalid verdict counts, and
win/tie/loss decompositions. It currently aggregates Arena-Elo by category,
not language. Prepare and evaluate each language separately for per-language
scores. Scores for this dataset are not directly comparable to the original
500-question English benchmark because the prompts and baselines differ.

This integration is experimental and inherits `verified: false`; model baselining
requires language-specific baseline answers and configured policy/judge endpoints.
