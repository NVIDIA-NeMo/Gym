# GraphWalks benchmark

Benchmark wrapper over the [`graphwalks` resources server](../../resources_servers/graphwalks/README.md)
for the [openai/graphwalks](https://huggingface.co/datasets/openai/graphwalks) dataset.

Each task supplies an adjacency list and asks the model to either list
the parents of a node (`problem_type: parents`) or return the BFS
frontier at exactly depth N (`problem_type: bfs`). Scoring is F1 over
the predicted node set vs. the expected node set, gated on the model
producing a `Final Answer: [...]` line.

## Variants

Two preset configs ship alongside this benchmark. Both apply the same
data + Skills prompt fixes (BFS depth disambiguation, self-parent
removal); they differ only in the tokenizer used for the `n_tokens`
column and an optional length filter.

| Variant | Config | Prepare script | Tokenizer | Max tokens | Output |
|---|---|---|---|---|---|
| Default | `config.yaml` | `prepare.py` | `o200k_base` (tiktoken) | none (no filter) | `data/graphwalks_benchmark.jsonl` |
| N3 1M | `config_n3_1m.yaml` | `prepare_n3_1m.py` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (HF) | `1048576` | `data/graphwalks_n3_1m_benchmark.jsonl` |

The N3 1M variant requires HF auth for the gated NVIDIA repo
(`HF_TOKEN` env or `huggingface-cli login`).

## Prepare benchmark data

```bash
# Default (o200k_base, no filter)
ng_prepare_benchmark "+config_paths=[benchmarks/graphwalks/config.yaml]"

# N3 1M variant
ng_prepare_benchmark "+config_paths=[benchmarks/graphwalks/config_n3_1m.yaml]"
```

For one-off custom builds (different tokenizer / cap / output path),
invoke `prepare.py` directly:

```bash
python benchmarks/graphwalks/prepare.py \
    --tokenizer_name meta-llama/Llama-3.1-8B-Instruct \
    --max_context_tokens 131072 \
    --output_fpath benchmarks/graphwalks/data/graphwalks_llama_128k_benchmark.jsonl
```

## Start environment

```bash
ng_run "+config_paths=[benchmarks/graphwalks/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
# Default variant
ng_collect_rollouts \
    +agent_name=graphwalks_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/graphwalks/data/graphwalks_benchmark.jsonl \
    +output_jsonl_fpath=results/graphwalks_rollouts.jsonl \
    +num_repeats=4

# N3 1M variant
ng_collect_rollouts \
    +agent_name=graphwalks_n3_1m_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/graphwalks/data/graphwalks_n3_1m_benchmark.jsonl \
    +output_jsonl_fpath=results/graphwalks_n3_1m_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

`compute_metrics()` emits `pass@k/accuracy`, `pass@1[avg-of-k]/accuracy`
via `compute_pass_majority_metrics`, plus per-`problem_type` subset
breakdowns via `compute_subset_metrics(subset_key="problem_type")` —
stratified pass@k keys like `problem_type=parents/pass@4/accuracy` and
`problem_type=bfs/pass@4/accuracy`.

For reasoning models the vLLM server should be started with a
`--reasoning-parser` matching the model (e.g. `nano_v3` for Nemotron-3
or `deepseek_r1`) so that `<think>...</think>` blocks are stripped
upstream of `Final Answer:` parsing.
