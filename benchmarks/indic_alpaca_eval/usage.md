# Indic AlpacaEval usage

Run these commands from the repository root with Gym and
[requirements.txt](requirements.txt) installed. Configure the
[judge endpoint](../../resources_servers/alpaca_pointwise_judge/README.md#configuring-the-judge)
before collecting rollouts.

## Prepare data

Prepare all 12 languages from the pinned AI4Bharat dataset:

```bash
gym eval prepare --benchmark indic_alpaca_eval
```

To use a local snapshot or select languages, call the preparer directly:

```bash
python -m benchmarks.indic_alpaca_eval.prepare \
  --dataset-dir /path/to/dataset \
  --languages hi ta
```

The snapshot contains `eval.parquet`, with `id`, `position`, `instruction`, and
`instruction_<Language>_translation` columns. Local snapshots with separate
`data/en/eval.parquet` and `data/<language>/eval.parquet` files are also
supported; their row IDs and indices must match the English file exactly.
Review flags from those files are preserved without filtering.

Every source must contain the full 805 aligned rows. Use
`--limit-per-language 2` to select a smaller evaluation after validation, and
`--output-dir` to keep it separate from a full preparation.

| File | Contents |
| --- | --- |
| `data/indic_alpaca_eval_benchmark.jsonl` | All selected tasks. |
| `data/<language>.jsonl` | Language shards sharing the full selection's coverage contract. |
| `data/smoke.jsonl` | One instruction per selected language, with its own coverage contract. |
| `data/manifest.json` | Source revision, hashes, language counts, evaluation-set ID, and filtering policy. |

The manifest records hashes of the source, adapter, rubric, and prepared data.
Local snapshots record their path and hashes without claiming the pinned
Hugging Face revision. Preserve the preparation manifest and resolved run
configuration with the results.

## Candidate settings

Prepared requests use these defaults:

| Setting | Value |
| --- | --- |
| Temperature | 1 |
| Top-p | 0.95 |
| Seed | 42 |
| Thinking | Enabled |
| Maximum output tokens | 120,000 |

These settings belong to each Alpaca task. Explicit run-level settings take
precedence, and other benchmarks retain their own defaults in combined runs.
Choose an output budget that leaves room for the input within the model's
context window. Configure the candidate endpoint's reasoning parser for the
served model. The judge receives only the final answer.

```bash
gym eval run --benchmark indic_alpaca_eval \
  --model-type vllm_model --model YOUR_MODEL \
  --model-url http://127.0.0.1:8000/v1 --model-api-key EMPTY \
  --temperature 1 --top-p 0.95 --max-output-tokens 4096 \
  --num-repeats 1 \
  --output results/indic_alpaca_eval/run/rollouts.jsonl
```

Gym writes metrics and failed measurements beside the rollouts. Successful
rows can resume from cache with `--resume`. Use a fresh output path when
changing the candidate, judge, prompt, or sampling settings.

## Gemma recipe

[gemma.yaml](gemma.yaml) configures a served `google/gemma-4-31B-it` model with
top-k 64 and the vLLM reasoning parser. Provider-specific sampling parameters
are passed through `metadata.extra_body`.

[run.py](run.py) uses Gym's `RunHelper` to start Gym servers around an existing
candidate endpoint, collect and aggregate rollouts, and shut down the servers.
It accepts a private file for the judge API key:

```bash
python -m benchmarks.indic_alpaca_eval.run \
  --judge-key-file /path/to/private/api-key \
  +input_jsonl_fpath=benchmarks/indic_alpaca_eval/data/smoke.jsonl \
  +output_jsonl_fpath=results/indic_alpaca_eval/gemma-smoke/rollouts.jsonl \
  +policy_base_url=http://127.0.0.1:8000/v1
```

Omit `--judge-key-file` when `INDIC_ALPACA_JUDGE_API_KEY` is already set.
Keep credentials out of CLI overrides. Use `--config` to select another Gym
recipe. The default full-run output is
`results/indic_alpaca_eval/gemma31b-it/rollouts.jsonl`.

## One language per job

Prepare all desired languages together, then use a generated language shard
as each job's input. For example, collect Hindi against running Gym servers:

```bash
gym eval run --no-serve \
  --agent indic_alpaca_eval_simple_agent \
  --input benchmarks/indic_alpaca_eval/data/hi.jsonl \
  --output results/indic_alpaca_eval/languages/hi/rollouts.jsonl \
  --num-repeats 1
```

Give every job its own Gym servers and output directory. Keep the candidate,
judge, sampling, and repeat settings identical across languages. The shared
evaluation-set ID and globally unique task indices allow Gym to combine the
outputs:

```bash
gym eval aggregate \
  --input-glob 'results/indic_alpaca_eval/languages/*/rollouts.jsonl' \
  --output results/indic_alpaca_eval/combined/rollouts.jsonl
```

Aggregation rejects mixed evaluation sets, duplicate task identities, and
inconsistent expected counts. The multilingual score requires all expected
tasks to have valid judgments. A single language shard retains the full
selection's coverage requirement.

## Smoke validation

Collect a small run from the prepared smoke dataset:

```bash
gym eval run --benchmark indic_alpaca_eval \
  --model-type vllm_model --model YOUR_MODEL \
  --model-url http://127.0.0.1:8000/v1 --model-api-key EMPTY \
  --input benchmarks/indic_alpaca_eval/data/smoke.jsonl \
  --output results/indic_alpaca_eval/smoke/rollouts.jsonl \
  --max-output-tokens 4096 --num-repeats 1

python -m benchmarks.indic_alpaca_eval.check_run \
  --inputs benchmarks/indic_alpaca_eval/data/smoke.jsonl \
  --rollouts results/indic_alpaca_eval/smoke/rollouts.jsonl \
  --max-output-tokens 4096 --smoke
```

The checker validates exact task coverage, unchanged metadata, score schema,
normalized reward, final-answer extraction, and the candidate output budget.
It expects one rollout per task. Pass `--max-output-tokens` when the run
overrides the prepared budget.

Run the preparation and resources-server checks with:

```bash
pytest tests/unit_tests/test_indic_alpaca_eval.py
gym env test +entrypoint=resources_servers/alpaca_pointwise_judge +should_validate_data=true
```
