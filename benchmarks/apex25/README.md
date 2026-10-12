# APEX 2025

Math problems from [MathArena](https://matharena.ai/?comp=apex--apex_2025)'s
APEX 2025 finals, sourced from `MathArena/apex_2025` on HuggingFace (12
problems). Companion to the larger `apex_shortlist` benchmark.

## Verification

Uses `math_with_judge` with **`should_use_judge: false`**, matching APEX
Shortlist. The final response must contain a nonempty, complete `\boxed{...}`.
Missing or empty boxes receive zero. `math-verify` checks symbolic equivalence;
there is no LLM fallback and no judge service or judge credentials are required.

This follows MathArena's symbolic-only grading approach. Gym uses `math-verify`
rather than MathArena's parser, so formatting and parsing differences can still
affect scores.

## Prompt

Byte-aligned with MathArena's own APEX prompt
(`configs/competitions/apex/apex_2025.yaml`):

```
Put your final answer within \boxed{}.

<question>
```

## Data preparation

```bash
gym eval prepare --benchmark apex25
```

Writes `data/apex25_benchmark.jsonl` with one row per problem:
`{"question": "...", "expected_answer": "..."}`. The HuggingFace dataset
revision is pinned in `prepare.py` (`HF_REVISION`) for reproducibility.

## Running servers

```bash
gym env start \
    --model-type inference_provider \
    --benchmark apex25
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent apex25_math_with_judge_simple_agent \
    --input benchmarks/apex25/data/apex25_benchmark.jsonl \
    --output results/apex25_rollouts.jsonl \
    --num-repeats 32
```

The example supplies all repeats at collection time; do not also repeat the
prepared dataset.

With only 12 problems the per-run variance is high — use several repeats
(`--num-repeats`) and report `avg@k`.
