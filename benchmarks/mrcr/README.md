# MRCR benchmark

Benchmark wrapper over the `mrcr` resources server for the
[openai/mrcr](https://huggingface.co/datasets/openai/mrcr) dataset.

## Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mrcr/config.yaml]"
```

Downloads the openai/mrcr HF dataset, token-counts each sample with
`tiktoken o200k_base`, and writes a filtered JSONL at
`benchmarks/mrcr/data/mrcr_benchmark.jsonl`. Samples over 200000 input
tokens are dropped to leave headroom for a model's own tokenizer to stay
under a 262144-token native context (some model tokenizers produce ~7-10%
more tokens than tiktoken `o200k_base`).

## Rollouts

```bash
ng_collect_rollouts \
    "+config_paths=[benchmarks/mrcr/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +agent_name=mrcr_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/mrcr/data/mrcr_benchmark.jsonl \
    +num_repeats=4
```

**Reasoning models**: start the vLLM server with a reasoning parser so the
`<think>...</think>` preamble is stripped before reaching the response body.
The MRCR grader checks that the response starts with a random prefix — if
reasoning leaks into `message.content`, every sample scores 0. For
Nemotron-3: `--reasoning-parser deepseek_r1`.

**Rollout count per task**: `num_repeats` for `type: benchmark` datasets is
a CLI flag (`+num_repeats=N`), not a YAML field — the YAML field only
applies to `type: train`/`type: validation` materialization. See
`+num_repeats_add_seed`: **do not set** it alongside `vllm_model`; the
Responses API schema rejects the extra `seed` field and every rollout
fails pydantic validation. Temperature sampling alone gives enough variance
across repeats.
