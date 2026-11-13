# Additional usage

Setup options for [Indic Arena Hard v2](README.md). Run commands from the
repository root in your Gym environment.

## Local data and language selection

The public Hub dataset needs no token. Use `HF_HOME` to select its cache
location, or prepare offline from a local download containing `test.parquet`:

```bash
python benchmarks/indic/indic-arena-hard-v2/prepare.py \
    --dataset-dir /path/to/indic-arena-hard-v2 \
    --languages hi ta
```

Omit `--dataset-dir` to use the pinned Hub revision. Omit `--languages` to
include all available languages, including the English control; use
`--languages en` for English only. Local mode also accepts
`data/<language>/test.parquet` and does not enforce the Hub revision.
Blank selected translations fail preparation; English is never substituted.

The local source can also be passed through Gym:

```bash
gym eval prepare --benchmark indic/indic-arena-hard-v2 \
    +prepare_script_args.dataset_dir=/path/to/indic-arena-hard-v2
```

Preparation preserves `row_id`, `row_index`, `category`, and `subcategory`.
It writes `language_name` as the full name and `language`, `language_code`,
and `subset_for_metrics` as the language code. Unused translation columns
are omitted. Translation-review metadata is retained when present.

## Pairwise evaluation

The dataset contains no baseline answers. Supply answers for the exact
source prompts in each selected language. English baselines are not
substituted for translations. Each baseline JSONL row has this format:

```json
{"language": "hi", "row_id": "<source uid>", "generation": "<baseline answer>"}
```

Answers are joined by `(language, row_id)`. Missing, duplicate, or empty
baselines fail preparation before replacing the output.

```bash
# Prepare selected languages with matching baselines
python benchmarks/indic/indic-arena-hard-v2/prepare.py \
    --languages hi ta \
    --baseline-file /path/to/baselines.jsonl

# Start the benchmark in pairwise mode
gym env start \
    --model-type vllm_model \
    --benchmark indic/indic-arena-hard-v2 \
    ++indic_arena_hard_v2_judge_mode=pairwise
```

Then collect rollouts using the command in the [README](README.md#example-usage).
For per-language pairwise scores, prepare and evaluate one language at a
time. Re-running preparation without `--baseline-file` replaces the output
with baseline-free tasks for default pointwise scoring.

## Pointwise smoke run

[`pointwise_smoke.yaml`](pointwise_smoke.yaml) uses the same configured
model to generate and judge answers, through separate candidate and judge
instances. This checks the workflow; it does not provide independent
model-quality scores or override English model settings.

```bash
# Configure an OpenAI-compatible Chat Completions endpoint
export INDIC_ARENA_SMOKE_BASE_URL=https://your-endpoint.example/v1
export INDIC_ARENA_SMOKE_MODEL=your-model
read -rsp 'Smoke endpoint API key: ' INDIC_ARENA_SMOKE_API_KEY
export INDIC_ARENA_SMOKE_API_KEY

# Prepare Hindi prompts without baselines
python benchmarks/indic/indic-arena-hard-v2/prepare.py --languages hi

# Start servers and keep this terminal running
gym env start --config benchmarks/indic/indic-arena-hard-v2/pointwise_smoke.yaml
```

In a second terminal with the same Gym environment:

```bash
gym eval run --no-serve \
    --agent indic_arena_hard_v2_simple_agent \
    --input benchmarks/indic/indic-arena-hard-v2/data/indic_arena_hard_v2_benchmark.jsonl \
    --output results/indic_arena_hard_v2_pointwise.jsonl \
    --prompt-config benchmarks/prompts/generic/default.yaml \
    --limit 3 --num-repeats 1 --concurrency 1 \
    --max-output-tokens 4096 --temperature 0
```

Inspect the answers, judgments, and invalid-score count, then stop the
servers with Ctrl+C. Remove `--limit 3` to evaluate the prepared split.
Both candidate and judge default to 4,096 output tokens in this preset;
candidate request overrides do not change the judge's token limit.
