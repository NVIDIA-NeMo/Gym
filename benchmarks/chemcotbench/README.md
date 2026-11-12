# ChemCoTBench V2

## Measurement contract

Source: [fresnellll/ChemCoTBench-V2](https://huggingface.co/datasets/fresnellll/ChemCoTBench-V2), revision `f0bb2fb00c97cb3257294a639e28f960f2da157e`, and [upstream code](https://github.com/fresnellll/ChemCoTBench-V2) at `dcd35470de4096a1b10ee9ed6f072bcee983a9cc`. The test release spans molecular editing, optimization, understanding, and reaction prediction. Preparation retains the documented missing-option/truncated-record exclusions.

Prepared task count: **5,219**. Data license: MIT. Integration code is
Apache-2.0. The source license notice is included in [LICENSE-ChemCoTBench](../../resources_servers/chemcotbench/LICENSE-ChemCoTBench).

The prepared JSONL contains rendered model inputs and separate `verifier_metadata`.
Gold answers, source records, and judge references stay in verifier metadata and
are not added to generation messages. Preparation renders the benchmark-specific
prompts and answer formats described in the verifier README. The standard
`simple_agent` is used with one model step and no tools or sandbox.

Mean binary L1 outcome reward, with separate L2 and L3 diagnostics. L1/L3 are scoring layers on the same questions, not dataset subsets. `run_layer3: true` enables process diagnostics; there is no switch to disable upstream L2 computation. A combined L1/L3 pooled score is a custom analysis metric, not an official benchmark score. Rewards are in [0, 1], higher is better.
See the [verifier README](../../resources_servers/chemcotbench/README.md) for extraction,
per-task rules, known protocol differences, and dependencies.

## Prepare and validate

Run from the Gym checkout with its development environment active:

```bash
uv sync --extra dev
source .venv/bin/activate
gym eval prepare --benchmark chemcotbench
gym env validate chemcotbench
gym env test chemcotbench
gym env test --resources-server chemcotbench +should_validate_data=true
gym env publish chemcotbench
```

Preparation downloads the pinned sources and writes `benchmarks/chemcotbench/data/test.jsonl`.
The prepared data and local caches are gitignored; Hugging Face also uses its
standard download cache. No external evaluator checkout or preconverted export
is required. `gym env publish` runs local manifest, fixture, and discovery checks;
it does not upload data or push code. To reuse a converted Gym dataset, pass
`prepare(input_path=...)` or run `python -m benchmarks.chemcotbench.prepare --input PATH`.
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
gym eval run --benchmark chemcotbench --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --limit 5 --num-repeats 1 --concurrency 4 \
  --temperature 0 --max-output-tokens 16384 \
  --output results/chemcotbench/sample.jsonl
```

For a full evaluation with four repeats:

```bash
gym eval run --benchmark chemcotbench --model-type vllm_model \
  --config /absolute/path/to/model.yaml \
  --split benchmark --num-repeats 4 --concurrency 512 \
  --temperature 1 --top-p 1 --max-output-tokens 131072 \
  --output results/chemcotbench/rollouts.jsonl ++num_repeats_add_seed=true
```

This is an example evaluation protocol, not a required baseline. Record the model,
judge, source revision, sampling, token budget, repeat count, and explicit seeds
for every comparison. Four repeats with seed addition request seeds 0–3 through the `vllm_model`
adapter; whether they are honored depends on the endpoint. Inspect failure
sidecars, judge parsing diagnostics, and response truncation as well as rewards.
A five-question sample is a runtime check, not a benchmark baseline.

No LLM judge is required.

ChemCoTBench installs a separate Python 3.11 PyTDC runtime for its legacy oracle
models. Its ordinary server uses the repository Python version. Startup validates
three pinned oracle checksums before loading pickles and checks finite predictions.
Missing artifacts are downloaded atomically; corrupt existing files fail with an
explicit path and are preserved for diagnosis. `timeout_seconds` defaults to 120
per answer (excluding queue time), with up to `max_concurrency: 4` reusable scoring
processes across both runtimes. Workers cache imports and Layer 3 reference data.
Rollout concurrency 512 does not change this scorer limit; set
`++chemcotbench.resources_servers.chemcotbench.max_concurrency=8`, for example,
when the host has enough CPU and memory. See the verifier README for worker
lifecycle and the exact yield/temperature and condition-ranking reward rules.
The fixture exercises the real
pinned non-MolOpt scorer; component tests also exercise the real MolOpt runtime.

The manifest remains `experimental: true` and the server config `verified: false`;
validation and example rollouts do not constitute certification.

## Independent integration

Preparation and helpers live in `benchmarks/chemcotbench/`; the verifier, component tests,
and preparation tests live in `resources_servers/chemcotbench/`. This integration does
not import or require another chemistry benchmark. It uses Gym core and the
dependencies and upstream sources documented above.

## Local validation evidence

On 2026-10-05, `gym eval prepare --benchmark chemcotbench` rebuilt all
5,219 rows from the pinned sources (using source download caches). The records
matched the previous prepared snapshot. `gym env publish chemcotbench` passed.
Five representative prepared questions were evaluated with the NVIDIA Luna
configuration above, temperature 0, one repeat, concurrency 4, and a 16,384-token
output budget. All five produced scores without verifier errors or truncation.
This is integration smoke evidence, not reproduction of a published full score.
