# LiveCodeBench

[LiveCodeBench](https://livecodebench.github.io/) is a benchmark for evaluating LLMs on competitive programming problems sourced from recent programming contests.

## Configuration

This benchmark uses the `code_gen` resource server with the `code_gen_simple_agent`.

- **Date range**: 2024-07-01 to 2025-02-01 (v5)
- **num_repeats**: 10
- **Sampling**: temperature=1.0, max_output_tokens=16384

## Usage

```bash
# Prepare data
ng_prepare_data +benchmark=livecodebench

# Start servers
ng_run +benchmark=livecodebench \
    "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts
ng_collect_rollouts +benchmark=livecodebench \
    "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_jsonl_fpath=results/livecodebench.jsonl
```
