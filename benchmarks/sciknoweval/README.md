# SciKnowEval V2

## Measurement contract

Source: [hicai-zju/SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval), revision `92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2`, `data/v2/sciknoweval_test_v2.jsonl`. All Biology, Chemistry, Material, and Physics questions at L1–L5 are included.

Prepared task count: **28,392**. Data license: MIT. Judge rubrics are downloaded at preparation time from pinned upstream source (`53addee640092d667439e9a8901bf55f2d92c9ed`) with SHA-256 validation; the upstream repository has no standalone code license notice. Integration code is
Apache-2.0. Downloaded materials retain their source licenses.

The prepared JSONL contains rendered model inputs and separate `verifier_metadata`.
Gold answers, source records, and judge references stay in verifier metadata and
are not added to generation messages. Preparation renders the benchmark-specific
prompts and answer formats described in the verifier README. The standard
`simple_agent` is used with one model step and no tools or sandbox.

`overall_score` averages repeats within each question, then questions equally. Per-level/domain/type metrics and a separate level macro are also available. Per-level scores also appear as `mean/L1`–`mean/L5` in the opening mean block and headline metrics, including only levels present in the scored data. Structured tasks use exact grading; open-ended and relation-extraction tasks use normalized model judgments. The Physics rubric correction and two relation-extraction judge substitutions differ from the published upstream protocol. Rewards are in [0, 1], higher is better.
See the [verifier README](../../resources_servers/sciknoweval/README.md) for extraction,
per-task rules, known protocol differences, and dependencies.

## Prepare and validate

Run from the Gym checkout with its development environment active:

```bash
uv sync --extra dev
source .venv/bin/activate
gym eval prepare --benchmark sciknoweval
gym env validate sciknoweval
gym env test sciknoweval
gym env test --resources-server sciknoweval +should_validate_data=true
gym env publish sciknoweval
```

Preparation downloads the pinned sources and writes `benchmarks/sciknoweval/data/test.jsonl`.
The prepared data and local caches are gitignored; Hugging Face also uses its
standard download cache. No external evaluator checkout or preconverted export
is required. `gym env publish` runs local manifest, fixture, and discovery checks;
it does not upload data or push code. To reuse a converted Gym dataset, pass
`prepare(input_path=...)` or run `python -m benchmarks.sciknoweval.prepare --input PATH`.
Both paths validate metadata, nonempty prompts, and unique IDs before atomically
replacing the output; failed preparation preserves existing output.

## Evaluate

Save this model configuration outside the checkout (for example, at
`/absolute/path/to/model.yaml`). Set `NVIDIA_API_KEY` in your environment; do not
put its value in a tracked YAML file. This example uses the NVIDIA-hosted Luna
Chat Completions endpoint through Gym's `vllm_model` adapter:

```yaml
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: ${oc.env:NVIDIA_API_KEY}
policy_model_name: openai/openai/gpt-5.6-luna
policy_model:
  responses_api_models:
    vllm_model:
      uses_interleaved_reasoning: false
      return_token_id_information: false
```

For another Chat Completions provider, replace the base URL, model name, and
credential reference. The model must support the requested output-token budget.
A five-question sample from the prepared dataset:

```bash
gym eval run --benchmark sciknoweval --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --limit 5 --num-repeats 1 --concurrency 4 \
  --temperature 0 --max-output-tokens 16384 \
  --output results/sciknoweval/sample.jsonl
```

For a full evaluation with four repeats:

```bash
gym eval run --benchmark sciknoweval --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --num-repeats 4 --concurrency 512 \
  --temperature 1 --top-p 1 --max-output-tokens 131072 \
  --output results/sciknoweval/rollouts.jsonl ++num_repeats_add_seed=true
```

This is an example evaluation protocol, not a required baseline. Record the model,
judge, source revision, sampling, token budget, repeat count, and explicit seeds
for every comparison. Four repeats with seed addition request seeds 0–3 through the `vllm_model`
adapter; whether they are honored depends on the endpoint. Inspect failure
sidecars, judge parsing diagnostics, and response truncation as well as rewards.
A five-question sample is a runtime check, not a benchmark baseline.

The benchmark config defaults to the policy model as judge. To use a separate
Responses-compatible judge, add `--config resources_servers/sciknoweval/configs/judge_model.yaml`
and configure `judge_base_url`, `judge_api_key`, and `judge_model_name` privately.
An OpenAI-compatible Chat Completions judge can instead be wired through Gym's
`vllm_model` adapter. The rubric stays benchmark-specific. Judge temperature,
output budget, and concurrency are explicit config fields; the benchmark config
uses temperature 0 and 16,384 output tokens by default.

The manifest remains `experimental: true` and the server config `verified: false`;
validation and example rollouts do not constitute certification.

## Independent integration

Preparation and helpers live in `benchmarks/sciknoweval/`; the verifier, component tests,
and preparation tests live in `resources_servers/sciknoweval/`. This integration does
not import or require another chemistry benchmark. It uses Gym core and the
dependencies and upstream sources documented above.

## Local validation evidence

On 2026-10-05, `gym eval prepare --benchmark sciknoweval` rebuilt all
28,392 rows from the pinned sources (using source download caches). The records
matched the previous prepared snapshot. `gym env publish sciknoweval` passed.
Five representative prepared questions were evaluated with the NVIDIA Luna
configuration above, temperature 0, one repeat, concurrency 4, and a 16,384-token
output budget. All five produced scores without verifier errors or truncation.
The 2 required judge responses were parsed successfully using the same model.
This is integration smoke evidence, not reproduction of a published full score.
