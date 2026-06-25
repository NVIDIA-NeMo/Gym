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
gym eval prepare --benchmark hendrycks_math

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark hendrycks_math

# Collecting rollouts
gym eval run --no-serve \
    --agent hendrycks_math_math_with_judge_simple_agent \
    --input benchmarks/hendrycks_math/data/hendrycks_math_benchmark.jsonl \
    --output results/hendrycks_math_rollouts.jsonl \
    --num-repeats 4
```
