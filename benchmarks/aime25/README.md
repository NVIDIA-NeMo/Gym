# AIME 2025

[AIME 2025](https://artofproblemsolving.com/wiki/index.php/2025_AIME) (American Invitational Mathematics Examination) is a math competition benchmark.

## Configuration

This benchmark uses the `math_with_judge` resource server with the `math_with_judge_simple_agent`.

- **num_repeats**: 32
- **Sampling**: temperature=1.0, max_output_tokens=16384

## Usage

```bash
# Prepare data
ng_prepare_data +benchmark=aime25

# Start servers
ng_run +benchmark=aime25 \
    "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts
ng_collect_rollouts +benchmark=aime25 \
    "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_jsonl_fpath=results/aime25.jsonl
```
