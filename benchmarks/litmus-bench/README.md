# Litmus-Bench v0.1

[Litmus-Bench v0.1](https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.1)
evaluates direct-answer molecular-property reasoning on 482 held-out questions.
It uses the reusable [`litmus_agent`](../../resources_servers/litmus_agent/README.md)
verifier; see that documentation for answer extraction and scoring behavior.

The pinned v0.1 test split contains no tool-use questions. Its prompts already
encode either boxed or double-parentheses answers through `use_box_format`, so
the benchmark applies no additional prompt template. The matching Hugging Face
train split is registered by the `litmus_agent` environment configuration.

## Prepare

```bash
gym eval prepare --benchmark litmus-bench
```

This downloads the pinned test release, validates its 482 rows, removes the
obsolete source `agent_ref`, and writes the gitignored artifact to
`benchmarks/litmus-bench/data/litmus-bench_benchmark.jsonl`.

## Evaluate

```bash
gym eval run \
  --benchmark litmus-bench \
  --model-type vllm_model \
  --split benchmark \
  --output results/litmus-bench.jsonl
```

The benchmark config uses five rollouts per question. Override `--num-repeats`
only when intentionally changing the evaluation protocol. It sets
`max_output_tokens` to 131,072 as an upper bound for long reasoning traces; the
effective generation budget is also limited by the model context window minus
the prompt. Use `--max-output-tokens` to select a smaller budget for a run.

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## ADME Tier-5 runs

The `adme_tier5_{direct,analogue,comparison}_sub10` exports use the same
`litmus_agent` verifier. They contain Gym-formatted prompts, answer-extraction
regexes, and per-row scoring tolerances, so no domain-specific resources server
or agent is needed. `prepare_adme_tier5.py` validates the rows, removes stale
`agent_ref` values, and materializes one benchmark artifact at a time.

Ten-question examples for each category are committed under `data/examples`.
For larger splits, set `ADME_TIER5_SOURCE_DIR` to the directory containing the
exported JSONL files. Select the source split with `ADME_TIER5_SPLIT`
(`validation` by default).

Prepare the committed examples through Gym:

```bash
ADME_TIER5_SPLIT=example gym eval prepare --config benchmarks/litmus-bench/config_adme_direct.yaml
ADME_TIER5_SPLIT=example gym eval prepare --config benchmarks/litmus-bench/config_adme_analogue.yaml
ADME_TIER5_SPLIT=example gym eval prepare --config benchmarks/litmus-bench/config_adme_comparison.yaml
```

Running `ADME_TIER5_SPLIT=example python
benchmarks/litmus-bench/prepare_adme_tier5.py` directly prepares all three.

### NVIDIA inference gateway

The gateway exposes Chat Completions, so use Gym's `vllm_model` bridge. Keep the
credential in the environment and let OmegaConf resolve it at runtime so the
secret is not exposed in the process command line:

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY first}"

gym eval run \
  --config benchmarks/litmus-bench/config_adme_direct.yaml \
  --model-type vllm_model \
  --model-url https://inference-api.nvidia.com \
  --model azure/openai/gpt-5.5 \
  --split benchmark \
  --output results/adme-tier5-direct_gpt55_rollouts.jsonl \
  --concurrency 8 \
  --max-output-tokens 8192 \
  '++policy_api_key=${oc.env:NVIDIA_API_KEY}' \
  ++policy_model.responses_api_models.vllm_model.extra_body.reasoning_effort=medium
```

Each config already requests five rollouts per question; do not also pass
`--num-repeats 5`, which would multiply the protocol to 25. Repeat with the
analogue and comparison configs. A single `gym eval run` starts the environment,
collects rollouts, and scores them.

Reward profiling consumes the materialized inputs written beside the rollout
file, not the original prepared benchmark file:

```bash
gym eval profile \
  --inputs results/adme-tier5-direct_gpt55_rollouts_materialized_inputs.jsonl \
  --rollouts results/adme-tier5-direct_gpt55_rollouts.jsonl
```

Use multiple repeats for meaningful per-task reward variance.

## Paired Tier 1/2 example

`config_tier12.yaml` provides a small paired tool-vs-no-tool Litmus benchmark
with ten unique questions: five Tier 1 and five Tier 2. Every question appears
once as `direct` and once as `mcp-python`; the prompt, expected answer,
extraction contract, and pair identifier are identical between arms. The tool
arm advertises `stateful_python_code_exec`.

Prepare the committed example through Gym:

```bash
gym eval prepare --config benchmarks/litmus-bench/config_tier12.yaml
```

The tool arm uses OpenSandbox with an RDKit-capable image. Export the
OpenSandbox and NVIDIA inference credentials before running:

```bash
export OPENSANDBOX_DOMAIN="<opensandbox-domain>"
export OPENSANDBOX_API_KEY="<opensandbox-key>"
export NVIDIA_API_KEY="<NVIDIA-inference-key>"

: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY}"
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY}"

gym eval run \
  --config benchmarks/litmus-bench/config_tier12.yaml \
  --model-type vllm_model \
  --model azure/openai/gpt-5.5 \
  --model-url https://inference-api.nvidia.com \
  --split benchmark \
  --output results/litmus-tier12-paired/rollouts.jsonl \
  --num-repeats "${NUM_REPEATS:-1}" \
  --concurrency "${CONCURRENCY:-1}" \
  --temperature 1 \
  --max-output-tokens 4096 \
  '++policy_api_key=${oc.env:NVIDIA_API_KEY}' \
  ++policy_model.responses_api_models.vllm_model.extra_body.reasoning_effort=medium
```

Set `--num-repeats` as needed for profiling. Direct rows do not create a
sandbox; tool sandboxes are created lazily.

Profile the paired results using the materialized inputs written by the run:

```bash
gym eval profile \
  --inputs results/litmus-tier12-paired/rollouts_materialized_inputs.jsonl \
  --rollouts results/litmus-tier12-paired/rollouts.jsonl
```
