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
gym eval prepare --benchmark critpt
```

This invokes `benchmarks/critpt/prepare.py` (declared as `prepare_script` in `config.yaml`),
which downloads the dataset and writes the full 70-problem flat-field JSONL to
`benchmarks/critpt/data/critpt_benchmark.jsonl` (gitignored).

## Run servers

```bash
gym env start --benchmark critpt --model-type vllm_model
```

While `gym env start` is up, the CritPt resources server exposes a `GET /status` endpoint
that reports live batch-fill progress (e.g. `{"pending_batches":[47],"batch_size":70}`).

## Collect rollouts

With `gym env start` already up, point `gym eval run --no-serve` at the flat-field JSONL and pass the
Turn 1 prompt config so the framework materializes `responses_create_params.input` at
rollout time.

```bash
gym eval run --no-serve \
    --agent critpt_benchmark_agent \
    --input benchmarks/critpt/data/critpt_benchmark.jsonl \
    --output results/critpt_rollouts.jsonl \
    --num-repeats 1 \
    --prompt-config benchmarks/critpt/prompts/turn1.yaml \
    --temperature 0.0
```

Use `temperature: 0.0` to match the nemo-skills baseline and ensure reproducible scores.

### One-shot alternative

Runs prepare + servers + rollout collection and tears the servers down afterwards:

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark critpt \
    --output results/benchmarks/critpt.jsonl \
    ++overwrite_metrics_conflicts=true \
    --split benchmark \
    --resume \
    ++reuse_existing_data_preparation=true \
    --model-url <your_endpoint> \
    --model-api-key <your_key> \
    --model <your_model> \
    --temperature 0.0
```

## Metrics

`pass@1/accuracy` is the headline metric.
