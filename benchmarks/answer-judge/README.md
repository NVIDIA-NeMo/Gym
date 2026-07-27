# Answer-Judge

Adds the `answer-judge` benchmark to Gym. Each row contains a math problem, a
predicted answer, an expected answer, and the gold `Judgement: Yes/No` label.

## Verification

This benchmark reuses `math_proof_judgement` because the verifier logic needed
here is the same deterministic `Judgement: Yes/No` parsing used by Skills'
`answer-judgement` metric.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark answer-judge

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark answer-judge

# Collecting rollouts
gym eval run --no-serve \
    --agent answer_judge_math_proof_judgement_simple_agent \
    --input benchmarks/answer-judge/data/answer-judge_benchmark.jsonl \
    --output results/answer-judge/rollouts.jsonl \
    --prompt-config benchmarks/prompts/judge/math.yaml
```
