# hendrycks_math

The Hendrycks MATH test split (5000 problems). Mirrors nemo-skills'
`nemo_skills/dataset/hendrycks_math` (which in turn sources the Qwen2.5-Math
GitHub-hosted preprocessing).

Data is fetched from the Qwen2.5-Math upstream at prepare time. `prepare.py`
applies Skills' renames (`answer` -> `expected_answer`, `question` ->
`problem`) and then further renames `problem` -> `question` for Gym.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/hendrycks_math/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/hendrycks_math/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=hendrycks_math_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/hendrycks_math/data/hendrycks_math_benchmark.jsonl \
    +output_jsonl_fpath=results/hendrycks_math_rollouts.jsonl \
    +num_repeats=4
```
