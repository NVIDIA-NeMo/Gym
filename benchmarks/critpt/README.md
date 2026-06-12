# CritPt Benchmark

Benchmark wrapper for [CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt), a
70-problem research-level physics benchmark. Each problem has a description and a Python
code template; the model must produce a precise numerical answer.

- **Tasks**: 70 physics problems
- **Reward**: aggregate accuracy from the [Artificial Analysis API](https://artificialanalysis.ai/documentation#critpt-api) (private test cases run server-side), distributed uniformly across the batch — every rollout shares the same float reward
- **Metrics**: `pass@1/accuracy` — fraction of problems the AA API accepts

The agent runs two LLM turns per problem:
1. **Turn 1** (`prompts/turn1.yaml`): step-by-step derivation ending in `Final Answer:`
2. **Turn 2**: populate the code template with the answer (the model sees its Turn 1 reasoning)

Turn 2's output is submitted to the AA API by `CritPtResourcesServer.verify()`.

## API key

The Artificial Analysis API key is read from `env.yaml`:

```yaml
artificial_analysis_api_key: <your-key>
```

The resources server config interpolates this via `${artificial_analysis_api_key}` — no key
in any committed file.

## Prepare benchmark data

`CritPt-Benchmark/CritPt` is a public HuggingFace dataset (no auth required).

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/critpt/config.yaml]"
```

This invokes `benchmarks/critpt/prepare.py` (declared as `prepare_script` in `config.yaml`),
which downloads the dataset and writes the full 70-problem flat-field JSONL to
`benchmarks/critpt/data/critpt_benchmark.jsonl` (gitignored).

## Run servers

```bash
ng_run "+config_paths=[benchmarks/critpt/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

While `ng_run` is up, the CritPt resources server exposes a `GET /status` endpoint
that reports live batch-fill progress (e.g. `{"pending_batches":[47],"batch_size":70}`).

## Collect rollouts

With `ng_run` already up, point `ng_collect_rollouts` at the flat-field JSONL and pass the
Turn 1 prompt config so the framework materializes `responses_create_params.input` at
rollout time.

```bash
ng_collect_rollouts \
    +agent_name=critpt_benchmark_agent \
    +input_jsonl_fpath=benchmarks/critpt/data/critpt_benchmark.jsonl \
    +output_jsonl_fpath=results/critpt_rollouts.jsonl \
    +num_repeats=1 \
    +prompt_config=benchmarks/critpt/prompts/turn1.yaml \
    "++responses_create_params={temperature: 0.0}"
```

Use `temperature: 0.0` to match the nemo-skills baseline and ensure reproducible scores.

### One-shot alternative

Runs prepare + servers + rollout collection and tears the servers down afterwards:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/critpt/config.yaml"

ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/benchmarks/critpt.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<your_endpoint> \
    ++policy_api_key=<your_key> \
    ++policy_model_name=<your_model> \
    "++responses_create_params={temperature: 0.0}"
```

## Metrics

`pass@1/accuracy` is the headline metric.
