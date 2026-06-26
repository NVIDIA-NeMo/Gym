# GSM8K

Grade school math word problems (test split, 1319 problems). Mirrors
nemo-skills' `nemo_skills/dataset/gsm8k`.

Data is fetched from the upstream openai/grade-school-math repo at
prepare time. `prepare.py` applies the same Skills transforms
(hardcoded answer fixes, `<<...>>` calc-string stripping, int-cast when
the expected answer is integer-valued), then renames `problem` ->
`question` for Gym.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark gsm8k

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark gsm8k

# Collecting rollouts
gym eval run --no-serve \
    --agent gsm8k_math_with_judge_simple_agent \
    --input benchmarks/gsm8k/data/gsm8k_benchmark.jsonl \
    --output results/gsm8k_rollouts.jsonl \
    --num-repeats 4
```
