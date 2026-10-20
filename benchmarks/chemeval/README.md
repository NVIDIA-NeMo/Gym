# ChemEval

## Measurement contract

Source: [Ooo1/ChemEval](https://huggingface.co/datasets/Ooo1/ChemEval), revision `61d82e727865e9c6c110fffd3ab920e0ab2edc42`, the zero-shot textual selection from `data/text-00000-of-00001.parquet`. It covers 53 tasks, 13 dimensions, and L1–L4; three-shot and multimodal questions are excluded. The nine judged tasks use bundled English V2 judge prompts; preparation does not download the original Chinese rubrics.

Prepared task count: **2,210**. Data license: The dataset card declares CC-BY-NC-4.0; the upstream repository declares CC-BY-NC-SA-4.0. Dataset files are downloaded at preparation time and remain gitignored. Bundled smoke examples are synthetic. Integration code is
Apache-2.0. Downloaded materials retain their source licenses.

The prepared JSONL contains rendered model inputs and separate `verifier_metadata`.
Gold answers, source records, and judge references stay in verifier metadata and
are not added to generation messages. Preparation renders the benchmark-specific
prompts and answer formats described in the verifier README. The standard
`simple_agent` is used with one model step and no tools or sandbox.

Use `overall_score`: average repeats per question, questions per task, then task scores equally. Generic Gym mean reward is question-weighted and differs. Fractional rewards, BIO token accuracy (including unquoted answer lists), and task-specific judge normalization are preserved. Rewards are in [0, 1], higher is better.
See the [verifier README](../../resources_servers/chemeval/README.md) for extraction,
per-task rules, known protocol differences, and dependencies.

## Prepare and validate

Run from the Gym checkout with its development environment active:

```bash
uv sync --extra dev
source .venv/bin/activate
gym eval prepare --benchmark chemeval
gym env validate chemeval
gym env test chemeval
gym env test --resources-server chemeval +should_validate_data=true
gym env publish chemeval
```

Preparation downloads the pinned sources and writes `benchmarks/chemeval/data/test.jsonl`.
The prepared data and local caches are gitignored; Hugging Face also uses its
standard download cache. No external evaluator checkout or preconverted export
is required. `gym env publish` runs local manifest, fixture, and discovery checks;
it does not upload data or push code. To reuse a converted Gym dataset, pass
`prepare(input_path=...)` or run `python -m benchmarks.chemeval.prepare --input PATH`.
Previously prepared judged rows with `judge_prefix`/`judge_suffix` must be regenerated
with `gym eval prepare --benchmark chemeval`; they are not accepted as V2 inputs.
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
gym eval run --benchmark chemeval --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --limit 5 --num-repeats 1 --concurrency 4 \
  --temperature 0 --max-output-tokens 16384 \
  --output results/chemeval/sample.jsonl
```

For a full evaluation with four repeats:

```bash
gym eval run --benchmark chemeval --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --num-repeats 4 --concurrency 512 \
  --temperature 1 --top-p 1 --max-output-tokens 131072 \
  --output results/chemeval/rollouts.jsonl ++num_repeats_add_seed=true
```

This is an example evaluation protocol, not a required baseline. Record the model,
judge, source revision, sampling, token budget, repeat count, and explicit seeds
for every comparison. Four repeats with seed addition request seeds 0–3 through the `vllm_model`
adapter; whether they are honored depends on the endpoint. Inspect failure
sidecars, judge parsing diagnostics, and response truncation as well as rewards.
A five-question sample is a runtime check, not a benchmark baseline.

The benchmark config defaults to the policy model as judge. To use a separate
Responses-compatible judge, add `--config resources_servers/chemeval/configs/judge_model.yaml`
and configure `judge_base_url`, `judge_api_key`, and `judge_model_name` privately.
An OpenAI-compatible Chat Completions judge can instead be wired through Gym's
`vllm_model` adapter. The rubric stays benchmark-specific. Judge temperature,
output budget, and concurrency are explicit config fields; the benchmark config
uses temperature 0 and 16,384 output tokens by default.

The manifest remains `experimental: true` and the server config `verified: false`;
validation and example rollouts do not constitute certification.

## Independent integration

Preparation and helpers live in `benchmarks/chemeval/`; the verifier, component tests,
and preparation tests live in `resources_servers/chemeval/`. This integration does
not import or require another chemistry benchmark. It uses Gym core and the
dependencies and upstream sources documented above.

## Local validation evidence

Eight English V2 rubrics are checked against the original module's system and
user messages. A separate regression test checks that molecular-description
questions use a property-description rubric instead of a naming rubric. Full preparation preserves all 2,210
generation inputs and all 1,760 deterministic rows; the 450 judged rows carry the
new protocol and question metadata. Unit tests cover both score scales, full
verdict retention, malformed judge output, and rejection of stale prepared inputs.

On 2026-10-05, `gym env publish chemeval` passed. Five synthetic examples ran
through NVIDIA-hosted Luna with temperature 0, one repeat, concurrency 4, and
16,384-token candidate and judge budgets. All five were scored without verifier
errors or truncation, and the English V2 judge response parsed successfully.
The bundled example rollouts contain this run; it is not a benchmark baseline.
