# Indic Arena Hard v2

Integrates [ai4bharat/indic-arena-hard-v2](https://huggingface.co/datasets/ai4bharat/indic-arena-hard-v2)
using Gym's `simple_agent`, model-server adapters, and generic prompt. The
Indic-specific `indic_arena_judge` resource server delegates pairwise judging to
the unchanged [Arena Hard v2 judge](../../../resources_servers/arena_judge/README.md)
and owns the optional pointwise rubric. Its default judge settings are temperature
0 and maximum 4,096 tokens. The English benchmark, judge, and their configurations
are unchanged; Indic mode and model settings are separate.

## Data

According to the [AI4Bharat dataset card](https://huggingface.co/datasets/ai4bharat/indic-arena-hard-v2/blob/aebdc931858cb6df854f16a0513e184fc38a3b1e/README.md),
Arena Hard v2 contains 750 source prompts, of which 504 are English. AI4Bharat
selected those 504 English prompts for translation into 14 Indian languages;
the remaining 246 prompts, originally written in other languages, are excluded.
The 504-question count therefore comes from selection by source language.
Each target language contains translations of the same 504 source questions.

Hub preparation pins dataset revision `aebdc931858cb6df854f16a0513e184fc38a3b1e`.
The dataset has one `default` configuration with a `test` split stored in
`test.parquet`. Each of its 504 rows contains the original English `prompt` and
14 translated columns such as `prompt_Hindi_translation`.

By default preparation includes the English control and all translations:
`en`, `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`, `te`,
and `ur`. Each language has 504 prompts, yielding 7,560 prepared tasks, including
7,056 translated tasks. Blank selected translations fail preparation; English
text is never substituted for a missing translation.

The selected prompt becomes `question`; the source `uid` becomes `row_id`, and
the output `uid` is `<language_code>:<row_id>` so translations remain distinct
tasks. `row_index` records the source row position. `language_name` holds the full
target-language name, while `language`, `language_code`, and `subset_for_metrics`
hold its code. Source `category` and `subcategory` are preserved: each language
has 363 `hard_prompt` tasks and 141 `creative_writing` tasks, which select their
respective judge prompts in pairwise mode. Unused translation columns are omitted
from prepared rows. The published snapshot contains no translation-review fields;
such fields are retained when present in older local inputs.

## Evaluation modes

Set `indic_arena_hard_v2_judge_mode` in the benchmark config, or override it with
`++indic_arena_hard_v2_judge_mode=pointwise` when starting an evaluation:

- `pairwise` (default): the English Arena Hard protocol, comparing against saved
  baseline answers with swapped positions and reporting Arena-Elo. Missing or blank
  baselines are rejected before calling the judge.
- `pointwise`: custom evaluation of one answer without any baseline. One judge call
  rates correctness, relevance, completeness, and language quality from 0 to 4 each.
  `pointwise/score` is their equally weighted average, rescaled to 0–100;
  rollout `reward` is that score divided by 100. This is not an Arena Hard win rate.

Pointwise metrics average repeats within each task, then average tasks. They also
include `pointwise/language/<code>/score` and each dimension's mean on its 0–4 scale.
An empty candidate answer scores zero without a judge call. Malformed judge output
gets reward zero and `invalid_pointwise: true`; it is excluded from rubric means and
counted in `pointwise/invalid_scores`. Check that count alongside the score. Failed
judge API calls go to Gym's failures sidecar rather than being scored as answers.

Data is read from a local download or downloaded at runtime; generated JSONL is
ignored by Git. Pass `--dataset-dir` to the downloaded dataset repository containing
`test.parquet` to prepare entirely offline, without a token. The older local layout,
`data/<language>/test.parquet`, remains supported. Local mode uses the supplied files
and discovers languages from their columns or language directories;
it does not enforce the Hub revision pin. The Hub dataset is public and needs no
token. Set `HF_HOME` to choose a Hugging Face cache directory.

## Baseline answers

This dataset contains **no baseline answers**. Its 504 English prompts form the
English subset of Arena Hard v2's 750 prompts; it is also distinct from the
500-question Arena Hard v0.1 evaluation set.
Supply answers generated for these exact prompts in each
language; the prepare script does not substitute English answers for translations.

The baseline JSONL format is:

```json
{"language": "hi", "row_id": "<source uid>", "generation": "<baseline answer>"}
```

Rows are joined by `(language, row_id)`. Missing, duplicate, and empty baseline
answers fail preparation before replacing the output. Without `--baseline-file`,
preparation warns and emits empty `baseline_answer` values, matching the existing
multilingual Arena Hard preparation workflow. That output can be used for pointwise
evaluation or prompt generation; pairwise scoring still requires baseline answers.

## Usage

Run from the repository root with the same Gym environment as the English benchmark.

The benchmark key is `indic/indic-arena-hard-v2`. Migrate commands from the previous
`indic_arena_hard` key and `benchmarks/indic_arena_hard/` directory to this path.
The agent is now `indic_arena_hard_v2_simple_agent`; the mode setting is
`indic_arena_hard_v2_judge_mode`. Set the `INDIC_ARENA_JUDGE_*` variables below for
Indic judging. English runs continue to use their existing `ARENA_JUDGE_*` variables.
Regenerate prepared data at the new path before running. The former
`gemini_pointwise.yaml` preset is now `pointwise_smoke.yaml`, with explicit endpoint,
model, and credential configuration.

```bash
# Prepare an existing local download (no network or token needed).
DATASET_DIR=/path/to/huggingface_datasets/ai4bharat/indic-arena-hard-v2
python benchmarks/indic/indic-arena-hard-v2/prepare.py --dataset-dir "$DATASET_DIR"

# The same local source can be passed through Gym.
gym eval prepare --benchmark indic/indic-arena-hard-v2 \
    +prepare_script_args.dataset_dir="$DATASET_DIR"

# Download all prompts for inspection (no baseline answers).
gym eval prepare --benchmark indic/indic-arena-hard-v2

# Prepare selected languages with baseline answers before judging.
python benchmarks/indic/indic-arena-hard-v2/prepare.py \
    --dataset-dir "$DATASET_DIR" \
    --languages hi ta \
    --baseline-file path/to/baselines.jsonl

# Configure INDIC_ARENA_JUDGE_BASE_URL, INDIC_ARENA_JUDGE_API_KEY and INDIC_ARENA_JUDGE_MODEL.
# Select the candidate model through Gym's usual model configuration.
gym env start --model-type vllm_model --benchmark indic/indic-arena-hard-v2

gym eval run --no-serve \
    --agent indic_arena_hard_v2_simple_agent \
    --input benchmarks/indic/indic-arena-hard-v2/data/indic_arena_hard_v2_benchmark.jsonl \
    --output results/indic_arena_hard_rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/prompts/generic/default.yaml
```

Omit `--languages` to prepare every language, including English, with a
complete baseline file. Use `--languages en` for the dataset's English control.
Re-running preparation without baselines replaces the output with prompts only.

## Pointwise smoke run

[`pointwise_smoke.yaml`](pointwise_smoke.yaml) uses a dedicated candidate model
instance and an Indic judge instance. Set `INDIC_ARENA_SMOKE_BASE_URL`,
`INDIC_ARENA_SMOKE_MODEL`, and `INDIC_ARENA_SMOKE_API_KEY` for an OpenAI-compatible
Chat Completions endpoint. The same model generates and judges answers in this
smoke preset. It provides workflow evidence, not independent model-quality scores.
These settings do not override the English benchmark's candidate or judge.

From the repository root, after activating the Gym environment:

```bash
# Enter a valid key interactively; it stays out of shell history and config files.
read -rsp 'Smoke endpoint API key: ' INDIC_ARENA_SMOKE_API_KEY
export INDIC_ARENA_SMOKE_API_KEY
export INDIC_ARENA_SMOKE_BASE_URL=https://your-endpoint.example/v1
export INDIC_ARENA_SMOKE_MODEL=your-model

# Prepare one language from the local download, without baseline answers.
python benchmarks/indic/indic-arena-hard-v2/prepare.py \
    --dataset-dir "$DATASET_DIR" --languages hi

# Start the servers in this terminal and keep them running.
gym env start --config benchmarks/indic/indic-arena-hard-v2/pointwise_smoke.yaml

# In a second terminal, activate the same Gym environment and run three tasks.
gym eval run --no-serve \
    --agent indic_arena_hard_v2_simple_agent \
    --input benchmarks/indic/indic-arena-hard-v2/data/indic_arena_hard_v2_benchmark.jsonl \
    --output results/indic_arena_hard_v2_pointwise.jsonl \
    --prompt-config benchmarks/prompts/generic/default.yaml \
    --limit 3 --num-repeats 1 --concurrency 1 \
    --max-output-tokens 4096 --temperature 0
```

Stop the servers with Ctrl+C in the first terminal when evaluation finishes.
The explicit input file requires `--no-serve`; the server-start command above
loads the smoke model and pointwise judge configuration.

The smoke candidate's request defaults and the judge use a 4,096-token output limit.
Explicit candidate request settings override the smoke defaults. The CLI flag above
changes the candidate limit; the judge limit is
`judge_chat_completions_create_params.max_tokens` on the resources server.
Remove `--limit 3` to evaluate the entire prepared split after inspecting smoke results.

## Metrics and validation status

In pairwise mode the shared [Arena judge](../../../resources_servers/indic_arena_judge/README.md) reports
`arena_elo/score`, bootstrap confidence bounds, invalid verdict counts, and
win/tie/loss decompositions. It currently aggregates Arena-Elo by category,
not language. Prepare and evaluate each language separately for per-language
scores. Scores for this dataset are not directly comparable to the original
500-question English benchmark because the prompts and baselines differ.

This integration is experimental and inherits `verified: false`. Both modes require
configured policy/judge endpoints; only pairwise mode requires baseline answers.

A live smoke test on September 23, 2026 used the earlier dataset source and three
Hindi prompts with Gemini as candidate and judge, one repeat, and concurrency one. This
predates the AI4Bharat source migration. All three answers completed within
the 4,096-token limit and produced valid pointwise judgments with no failed
rollouts. The self-judge assigned 100/100 to each; this validates the workflow,
not benchmark quality or comparability with Arena Hard win rates. Trajectory
capture was disabled, so Gym's trajectory health checks reported `unobserved`;
the saved answers, response status, judgments, and aggregate metrics were inspected
directly.
