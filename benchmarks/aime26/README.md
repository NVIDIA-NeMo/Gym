# AIME 2026

AIME 2026 (American Invitational Mathematics Examination) — 30 competition math problems requiring integer answers in [0, 999]. Verification uses symbolic math equivalence (`math_verify`) with an LLM judge fallback.

## Prepare data

```bash
gym eval prepare --benchmark aime26
```

## Run servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark aime26
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent aime26_math_with_judge_simple_agent \
    --input benchmarks/aime26/data/aime26_benchmark.jsonl \
    --output results/aime26_rollouts.jsonl \
    --num-repeats 4
```
