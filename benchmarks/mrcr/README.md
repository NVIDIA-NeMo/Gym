# MRCR benchmark

Benchmark wrapper over the [`mrcr` resources server](../../resources_servers/mrcr/README.md)
for the [openai/mrcr](https://huggingface.co/datasets/openai/mrcr) dataset.

Each task is a multi-turn conversation with a final-turn "prepend `<prefix>`
to the Nth occurrence and reproduce it exactly" instruction. Scoring:
`SequenceMatcher.ratio()` between stripped response and stripped expected
answer, gated on the response starting with the random prefix.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mrcr/config.yaml]"
```

Downloads the HF dataset, token-counts each sample with `tiktoken o200k_base`,
and writes `benchmarks/mrcr/data/mrcr_benchmark.jsonl`. Samples over 200000
input tokens are dropped to leave headroom for model-side tokenizers (which
can be 7–10% heavier than tiktoken) to stay under a 262144-token native
context.

## Start environment

```bash
ng_run "+config_paths=[benchmarks/mrcr/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=mrcr_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/mrcr/data/mrcr_benchmark.jsonl \
    +output_jsonl_fpath=results/mrcr_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

`compute_metrics()` emits `pass@k/accuracy`, `pass@1[avg-of-k]/accuracy`
via `compute_pass_majority_metrics`, plus per-`n_needles` subset breakdowns
via `compute_subset_metrics(subset_key="n_needles")` — stratified pass@k
keys like `n_needles=2/pass@4/accuracy`, `n_needles=4/pass@4/accuracy`,
`n_needles=8/pass@4/accuracy`.
