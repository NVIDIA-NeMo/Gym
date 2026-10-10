# Indic FrontierMath

Twelve math questions in English and fourteen Indic languages, with one response
per question and language. Accuracy is the fraction of correct answers. A complete
language has twelve questions; report coverage for incomplete runs.

## Prepare

Run from the Gym repository root:

```bash
python -m pip install -r benchmarks/indic/frontier_math/requirements.txt
gym eval prepare --benchmark indic/frontier_math
```

The loader finds `huggingface_datasets/anushakamathofficial/indic_frontiermath`
under an ancestor of the checkout. To specify a source directory or languages:

```bash
python benchmarks/indic/frontier_math/prepare.py \
  --dataset-dir /path/to/indic_frontiermath --languages en hi
```

Preparation writes raw rows for Gym's prompt system, rendered requests for
standalone runners, per-language JSONL files, and a preparation manifest under
`benchmarks/indic/frontier_math/data/`. Generated data is gitignored. The configured
prompt is applied exactly once; answer metadata is excluded from model input.

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
  --temperature 1 --max-output-tokens 240000

python benchmarks/indic/frontier_math/summarize.py \
  results/indic_frontiermath/en.jsonl
```

Configure top-p, top-k, and thinking in the model connection settings. Keep model
and generation settings consistent across languages. The summary reports strict
exact boxed-answer accuracy and auxiliary normalized accuracy, which only repairs
gold-independent formatting failures before exact re-grading. Inspect
`grading_status` for missing answers, parse failures, and verifier errors. No LLM
judge is used.

## Test

```bash
python -m pytest resources_servers/frontiermath/tests benchmarks/indic/frontier_math/tests
```
