# APEX 2025

Math problems from [MathArena](https://matharena.ai/?comp=apex--apex_2025)'s
APEX 2025 finals, sourced from `MathArena/apex_2025` on HuggingFace (12
problems). Companion to the larger `apex_shortlist` benchmark.

## Verification

Uses the `math_with_autograder` resource server: **symbolic-first with an
autograder LLM fallback**. The HuggingFace `math-verify` library checks symbolic
equivalence of the model's `\boxed{...}` against `expected_answer`; only on a
symbolic miss is the judge asked "is this answer Correct or Incorrect?". Answers
math-verify already accepts never reach the judge, so grading stays deterministic
for nearly every rollout.

The judge is a **dedicated** model (`judge_nemotron3ultra.yaml` →
Nemotron 3 Ultra by default), not the policy model, so the grading standard stays
fixed when comparing different policy models.

APEX answers are short numeric / fractional / radical values, which math-verify
handles well: graded against MathArena's own `correct` labels on their published
model outputs, symbolic-only already agrees on **99.9%** of rollouts with no
false positives. The autograder is used anyway so that all MathArena benchmarks
here (`apex25`, `arxivmath_*`) grade identically and stay mutually comparable —
and it recovers formatting-only misses such as `6\,266\,942\,768` for
`6266942768`. Scores may therefore run marginally above the MathArena
leaderboard, which grades symbolically only.

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
    --agent apex25_math_with_autograder_simple_agent \
    --input benchmarks/apex25/data/apex25_benchmark.jsonl \
    --output results/apex25_rollouts.jsonl \
    --num-repeats 4
```

The judge needs `NVIDIA_API_KEY` in the environment.

With only 12 problems the per-run variance is high — use several repeats
(`--num-repeats`) and report `avg@k`.
