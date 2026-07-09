# Math-500

Adds the `math-500` benchmark to Gym on top of the shared `math_with_judge`
resource server.

## Details

- Data source: OpenAI `prm800k/math_splits/test.jsonl`
- Evaluation: free-form math answer with symbolic verification
- Prompt: shared `generic_math` prompt mirroring Skills' `generic/math`
- Verification: symbolic-only (`should_use_judge: false`) to match Skills'
  default `eval_type=math`

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark math-500

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark math-500

# Collecting rollouts
gym eval run --no-serve \
    --agent math_500_math_with_judge_simple_agent \
    --input benchmarks/math-500/data/math-500_benchmark.jsonl \
    --output results/math-500/rollouts.jsonl \
    --prompt-config benchmarks/prompts/generic/math.yaml
```
