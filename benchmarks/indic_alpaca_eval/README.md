# Indic AlpacaEval pointwise

Evaluate 805 translated instructions per language: 9,660 tasks across Bengali,
Gujarati, Hindi, Kannada, Marathi, Malayalam, Nepali, Odia, Punjabi, Tamil,
Telugu, and Urdu. Canonical codes are `bn gu hi kn mr ml ne or pa ta te ur`.
Preparation accepts `ka`, `mar`, and `mal` as aliases for `kn`, `mr`, and `ml`.
English is used to check alignment and is not evaluated.

This is a **custom pointwise adaptation**, with overall and diagnostic scores
on a 1–5 scale. These scores are not official AlpacaEval pairwise or
length-controlled win rates. See the
[judge rubric and metrics](../../resources_servers/alpaca_pointwise_judge/README.md).

## Prepare data

Run from a checkout with Gym and [requirements.txt](requirements.txt) installed:

```bash
python -m benchmarks.indic_alpaca_eval.prepare
# Or supply a local snapshot:
python -m benchmarks.indic_alpaca_eval.prepare --dataset-dir /path/to/dataset
```

The default downloads
[anushakamathofficial/indic_alpaca_eval](https://huggingface.co/datasets/anushakamathofficial/indic_alpaca_eval)
at revision `5a173582c9e9f184e6a8fdefb0fb12ae485d56fa`.
`gym eval prepare --benchmark indic_alpaca_eval` uses the same preparer.
Use `--languages hi ta` and `--limit-per-language 2` for a smaller selection.

Preparation checks all 805 English row IDs and indices and exact ordered
correspondence in each selected translation. It rejects missing, duplicate,
reordered, mislabeled, or empty translated rows. Instructions are passed to
the policy unchanged. Translation review flags are retained without filtering;
alignment checks do not establish translation quality.

Generated files are ignored by Git:

- `data/indic_alpaca_eval_benchmark.jsonl`: the selected tasks.
- `data/bn.jsonl`, etc.: language shards with a shared coverage contract.
- `data/smoke.jsonl`: one instruction per selected language, with its own contract.
- `data/manifest.json`: source revision, source/adapter/rubric/output hashes,
  counts, evaluation-set ID, and filtering policy. Local snapshots record
  their path and hashes without claiming the pinned revision.

The external dataset declares CC-BY-NC-4.0. Benchmark text is downloaded at
runtime; only synthetic Apache-2.0 examples are included in this repository.

## Configure the judge and run

Set `INDIC_ALPACA_JUDGE_BASE_URL`, `INDIC_ALPACA_JUDGE_MODEL`, and
`INDIC_ALPACA_JUDGE_API_KEY` for your OpenAI-compatible judge endpoint. The judge
must support Chat Completions with JSON-schema structured output and the
configured reasoning effort. Its default settings are temperature 0,
4,096 maximum output tokens, low reasoning effort, and concurrency 1.
Use a configuration override for providers with different supported parameters.

Prepared candidate requests default to temperature 1, top-p 0.95, seed 42,
thinking enabled, and 120,000 maximum output tokens. The endpoint must have
enough context for the input plus this output budget; reduce the budget for
smaller context windows. These defaults belong to each Alpaca row, so they do
not replace another benchmark's sampling or metadata in a combined run.
Explicit run-level settings take precedence. Repeat seeds are independent.

```bash
gym eval run --benchmark indic_alpaca_eval \
  --model-type vllm_model --model YOUR_MODEL \
  --model-url http://127.0.0.1:8000/v1 --model-api-key EMPTY \
  --max-output-tokens 4096
```

For a served Gemma model, [gemma.yaml](gemma.yaml) is a single-benchmark recipe
that adds top-k 64 and configures the vLLM reasoning parser. Sampling travels
through request parameters, including `metadata.extra_body` for top-k.
The [run.py](run.py) launcher starts Gym's servers around an existing policy
endpoint using `RunHelper`, collects and aggregates rollouts, then stops Gym.

```bash
python -m benchmarks.indic_alpaca_eval.run \
  --judge-key-file /path/to/private/api-key \
  +input_jsonl_fpath=benchmarks/indic_alpaca_eval/data/smoke.jsonl \
  +output_jsonl_fpath=results/indic_alpaca_eval/smoke/rollouts.jsonl \
  +policy_base_url=http://127.0.0.1:8000/v1
```

Omit `--judge-key-file` when the key is already in the environment. Keep keys
out of CLI overrides. The default full-run output is
`results/indic_alpaca_eval/gemma31b-it/rollouts.jsonl`. Gym writes metrics and
failed measurements beside the rollouts; successful rows resume from cache.
Use a fresh output path when changing models, prompts, or sampling settings.

Language shards have globally unique task indices and can be collected
independently, then combined with `gym eval aggregate`. A single shard does
not count as a complete multilingual evaluation. Preserve the configuration
and preparation manifest with every run.

## Validate

```bash
pytest tests/unit_tests/test_indic_alpaca_eval.py
gym env test +entrypoint=resources_servers/alpaca_pointwise_judge +should_validate_data=true
python -m benchmarks.indic_alpaca_eval.check_run \
  --inputs benchmarks/indic_alpaca_eval/data/smoke.jsonl \
  --rollouts results/indic_alpaca_eval/smoke/rollouts.jsonl --smoke
```

If a run overrides the prepared token budget, pass that budget to the checker
with `--max-output-tokens`. The checker verifies exact task coverage, metadata,
score validity, normalized reward, final-answer extraction, and token budgets.
Tests cover alignment, provenance, CLI overrides, mixed-benchmark metadata,
repeat seeds, judge failures, and aggregation. The resources server remains
`verified: false` pending a reviewed baseline.
