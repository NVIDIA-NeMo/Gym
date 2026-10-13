# ChemBench

## Measurement contract

Source: [jablonkagroup/ChemBench](https://huggingface.co/datasets/jablonkagroup/ChemBench), revision `6e1d25748952393f44e35b8e85bbe567246a6430`, the `train` split across eight chemistry topics. The selected release contains 1,542 multiple-choice and 243 numeric questions; preference questions are excluded.

Prepared task count: **1,785**. Data license: MIT. Integration code is
Apache-2.0. The source license notice is included in [LICENSE-ChemBench](../../resources_servers/chembench/LICENSE-ChemBench).

The prepared JSONL contains rendered model inputs and separate `verifier_metadata`.
Gold answers, source records, and judge references stay in verifier metadata and
are not added to generation messages. Preparation renders the benchmark-specific
prompts and answer formats described in the verifier README. The standard
`simple_agent` is used with one model step and no tools or sandbox.

Mean binary reward. Multi-select questions require the complete answer set; numeric questions use the pinned upstream strict tolerance rule, including its zero/negative-target behavior. Extraction is deterministic and does not use the upstream LLM fallback. Rewards are in [0, 1], higher is better.
See the [verifier README](../../resources_servers/chembench/README.md) for extraction,
per-task rules, known protocol differences, and dependencies.

## Prepare and validate

Run from the Gym checkout with its development environment active:

```bash
uv sync --extra dev
source .venv/bin/activate
gym eval prepare --benchmark chembench
gym env validate chembench
gym env test chembench
gym env test --resources-server chembench +should_validate_data=true
gym env publish chembench
```

Preparation downloads the pinned sources and writes `benchmarks/chembench/data/test.jsonl`.
The prepared data and local caches are gitignored; Hugging Face also uses its
standard download cache. No external evaluator checkout or preconverted export
is required. `gym env publish` runs local manifest, fixture, and discovery checks;
it does not upload data or push code. To reuse a converted Gym dataset, pass
`prepare(input_path=...)` or run `python -m benchmarks.chembench.prepare --input PATH`.
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
gym eval run --benchmark chembench --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --limit 5 --num-repeats 1 --concurrency 4 \
  --temperature 0 --max-output-tokens 131072 \
  --output results/chembench/sample.jsonl
```

For a full evaluation with four repeats:

```bash
gym eval run --benchmark chembench --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --num-repeats 4 --concurrency 512 \
  --temperature 1 --top-p 1 --max-output-tokens 131072 \
  --output results/chembench/rollouts.jsonl ++num_repeats_add_seed=true
```

This is an example evaluation protocol, not a required baseline. Record the model,
judge, source revision, sampling, token budget, repeat count, and explicit seeds
for every comparison. Four repeats with seed addition request seeds 0–3 through the `vllm_model`
adapter; whether they are honored depends on the endpoint. Inspect failure
sidecars, judge parsing diagnostics, and response truncation as well as rewards.
A five-question sample is a runtime check, not a benchmark baseline.

No LLM judge is required.

The manifest remains `experimental: true` and the server config `verified: false`;
validation and example rollouts do not constitute certification.

## Independent integration

Preparation and helpers live in `benchmarks/chembench/`; the verifier, component tests,
and preparation tests live in `resources_servers/chembench/`. This integration does
not import or require another chemistry benchmark. It uses Gym core and the
dependencies and upstream sources documented above.

## Local validation evidence

On 2026-10-06, preparation rebuilt all 1,785 rows from pinned cached sources.
LaTeX cleanup changed 509 prompts; all prepared prompts matched the pinned
upstream default cleanup, with gold labels and numeric tolerances unchanged.
`gym env publish chembench` passed, and all 151 ChemBench tests passed.
The five committed examples were regenerated with the NVIDIA Luna configuration
above, temperature 0, one repeat, concurrency 4, and a 131,072-token output limit.
All five completed without truncation or verifier errors. Their inputs, rollouts,
and dataset statistics are saved in `resources_servers/chembench/data/`.
This is integration smoke evidence, not reproduction of a published full score.
