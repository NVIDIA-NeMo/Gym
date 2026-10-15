# Indic FrontierMath

Uses `ai4bharat/indic-frontier-math`: twelve public sample math questions in
English and fourteen Indic languages, with one response per question and language.
Accuracy is the fraction of correct answers. A complete language has twelve
questions; report coverage for incomplete runs.

## Prepare

Run from the Gym repository root:

```bash
python -m pip install -r benchmarks/indic/frontier_math/requirements.txt
gym eval prepare --benchmark indic/frontier_math
```

The loader downloads `sample.parquet` from `ai4bharat/indic-frontier-math` at
revision `dcae6631c88d3ae1a26c188eb8aa13ad4c7f4ad7`. Authenticate with Hugging Face
if access is required. For an offline cache, provide a directory containing that
file:

```bash
python benchmarks/indic/frontier_math/prepare.py \
  --dataset-dir /path/to/indic-frontier-math --languages en hi
```

The Gym preparer also accepts `+prepare_script_args.dataset_dir=/path/to/indic-frontier-math`.
By default, all fifteen languages are prepared. To prepare the twelve evaluation
languages:

```bash
python benchmarks/indic/frontier_math/prepare.py \
  --dataset-dir /path/to/indic-frontier-math \
  --languages bn gu hi kn mr ml ne or pa ta te ur
```

Each source row contains the English `problem`, `answer`, and one
`problem_<Language>_translation` column per Indic language. Preparation validates
problem IDs, English statement fingerprints, tiers, and answers against the
versioned answer key. Missing or empty selected translations fail preparation.
Source solutions are excluded from prepared rows and model input.

Preparation writes raw rows for Gym's prompt system, rendered requests for
standalone runners, per-language JSONL files, and a preparation manifest under
`benchmarks/indic/frontier_math/data/`. Generated data is gitignored. The configured
prompt is applied exactly once; answer metadata is excluded from model input.
The manifest records the source file's SHA-256 and the selected languages. Local
overrides record a null source revision because a directory alone cannot establish
which Hub revision its contents represent.

The source does not supply per-item translation-review labels. Translated rows
therefore use `judge_pass_stage=not_provided` and `human_evaluation_pending=true`;
no per-item approval is inferred from the dataset's aggregate review description.
This snapshot includes corrected translations, so regenerate prepared data and
label new results with the dataset revision or file checksum when comparing runs.

## Run

Start Gym against an existing model endpoint:

```bash
gym env start --benchmark indic/frontier_math --model-type vllm_model \
  --model-url http://HOST:PORT/v1 --model MODEL_NAME
```

In another terminal, run prepared requests through the simple agent and verifier:

```bash
gym eval run --no-serve \
  --agent indic_frontiermath_simple_agent \
  --input benchmarks/indic/frontier_math/data/en.jsonl \
  --output results/indic_frontiermath/en.jsonl \
  --num-repeats 1 --concurrency 1 \
  --temperature 1 --top-p 0.95 --max-output-tokens 240000

resources_servers/frontiermath/.venv/bin/python benchmarks/indic/frontier_math/summarize.py \
  results/indic_frontiermath/en.jsonl
```

Configure top-k and thinking in the model connection settings. Keep model
and generation settings consistent across languages. The summary reports strict
exact boxed-answer accuracy and auxiliary normalized accuracy, which only repairs
gold-independent formatting failures before exact re-grading. Inspect
`grading_status` for missing answers, parse failures, and verifier errors. No LLM
judge is used.

Equality recovery accepts only a single-letter answer assignment such as `n=214`.
It does not recover values from inequalities, chained equalities, or expressions
on the left-hand side such as `2n=214`.

Masked attempts and verifier errors or timeouts are excluded from both accuracy
calculations. Missing or mathematically invalid model answers still count as
incorrect. The summary retains total problem/attempt counts and grading statuses,
and reports scored, excluded, masked, and verifier-failure counts separately.
`complete_12_problem_coverage` requires a successful grading attempt for all twelve
problems; `observed_12_problem_coverage` includes failed attempts. Languages with
no successfully graded attempts have `null` accuracy.
Failures during auxiliary normalized re-grading also exclude the attempt from
both scores, keeping their denominators aligned, and appear in
`normalization_failure_attempts` and `normalization_failures`.

Run the summary in the resources server's virtual environment so symbolic
normalization uses the verifier's pinned dependencies.

## Test

```bash
python -m pytest tests/unit_tests/test_indic_frontier_math.py
gym env test +entrypoint=resources_servers/frontiermath +should_validate_data=true
```
