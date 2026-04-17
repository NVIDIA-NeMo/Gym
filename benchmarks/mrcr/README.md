# MRCR benchmark

Benchmark wrapper over the `mrcr` resources server for the
[openai/mrcr](https://huggingface.co/datasets/openai/mrcr) dataset.

## Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mrcr/config.yaml]"
```

Downloads the openai/mrcr HF dataset, token-counts each sample with
`tiktoken o200k_base`, and writes a filtered JSONL at
`benchmarks/mrcr/data/mrcr_benchmark.jsonl`. Samples over 262144 input
tokens are dropped (Nemotron-3-Super-120B context window).

## Rollouts

Typical command (from the submission host, via the
`nemo-skills-recipes/migrate-gym-mrcr/run_mrcr_gym.py` recipe):

```bash
ns nemo_gym_rollouts \
    "+config_paths=[benchmarks/mrcr/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +agent_name=mrcr_benchmark_simple_agent \
    input_file=benchmarks/mrcr/data/mrcr_benchmark.jsonl \
    ...
```

Rollout count per task is controlled by `num_repeats` in
`benchmarks/mrcr/config.yaml` (default 4 to match NeMo Skills
`benchmarks="mrcr:4"`). All repeats are dispatched through a single
multi-node vLLM with data-parallel.
