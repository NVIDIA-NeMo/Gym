# MRCR benchmark

Benchmark wrapper over the [`mrcr` resources server](../../resources_servers/mrcr/README.md)
for the [openai/mrcr](https://huggingface.co/datasets/openai/mrcr) dataset.

Each task is a multi-turn conversation with a final-turn "prepend `<prefix>`
to the Nth occurrence and reproduce it exactly" instruction. Scoring:
`SequenceMatcher.ratio()` between stripped response and stripped expected
answer, gated on the response starting with the random prefix.

## Variants

| Variant | Config | Prepare script | Tokenizer | Max tokens | Output |
|---|---|---|---|---|---|
| Default | `config.yaml` | `prepare.py` | `o200k_base` (tiktoken) | none (no filter) | `data/mrcr_benchmark.jsonl` |
| N3 128k | `config_n3_128k.yaml` | `prepare_n3_128k.py` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (HF) | `131072` | `data/mrcr_n3_128k_benchmark.jsonl` |
| N3 1M | `config_n3_1m.yaml` | `prepare_n3_1m.py` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (HF) | `1048576` | `data/mrcr_n3_1m_benchmark.jsonl` |

The N3 variants require HF auth for the gated NVIDIA repo
(`HF_TOKEN` env or `huggingface-cli login`).

For one-off custom builds (different tokenizer / cap / output path),
invoke `prepare.py` directly:

```bash
python benchmarks/mrcr/prepare.py \
    --tokenizer_name nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --max_context_tokens 131072 \
    --output_fpath benchmarks/mrcr/data/mrcr_n3_128k_benchmark.jsonl
```

## Prepare benchmark data

```bash
# Default (o200k_base, no filter)
ng_prepare_benchmark "+config_paths=[benchmarks/mrcr/config.yaml]"

# N3 128k variant
ng_prepare_benchmark "+config_paths=[benchmarks/mrcr/config_n3_128k.yaml]"

# N3 1M variant
ng_prepare_benchmark "+config_paths=[benchmarks/mrcr/config_n3_1m.yaml]"
```

## Start environment

```bash
ng_run "+config_paths=[benchmarks/mrcr/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
# Default variant
ng_collect_rollouts \
    +agent_name=mrcr_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/mrcr/data/mrcr_benchmark.jsonl \
    +output_jsonl_fpath=results/mrcr_rollouts.jsonl \
    +num_repeats=4

# N3 128k variant
ng_collect_rollouts \
    +agent_name=mrcr_n3_128k_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/mrcr/data/mrcr_n3_128k_benchmark.jsonl \
    +output_jsonl_fpath=results/mrcr_n3_128k_rollouts.jsonl \
    +num_repeats=4

# N3 1M variant
ng_collect_rollouts \
    +agent_name=mrcr_n3_1m_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/mrcr/data/mrcr_n3_1m_benchmark.jsonl \
    +output_jsonl_fpath=results/mrcr_n3_1m_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

`compute_metrics()` emits `pass@k/accuracy`, `pass@1[avg-of-k]/accuracy`
via `compute_pass_majority_metrics`, plus per-`n_needles` subset breakdowns
via `compute_subset_metrics(subset_key="n_needles")` — stratified pass@k
keys like `n_needles=2/pass@4/accuracy`, `n_needles=4/pass@4/accuracy`,
`n_needles=8/pass@4/accuracy`.
