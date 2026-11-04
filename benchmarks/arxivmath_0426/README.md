# ArXivMath 04/2026

Research-level final-answer math problems from
[MathArena](https://matharena.ai/arxivmath/)'s ArXivMath **04/2026** release,
sourced from `MathArena/arxivmath-0426` on HuggingFace (41 problems). Problems
are drawn from arXiv papers published that month; answers are a single numeric
value or a pure LaTeX expression.

MathArena publishes a new ArXivMath problem set each month and scores each
release on its own leaderboard, so every month is a separate Gym benchmark
(see also `arxivmath_0526`). Earlier releases (12/2025–03/2026) are marked
deprecated upstream due to contamination risk and are not mirrored here.

## Verification

Uses the `math_with_autograder` resource server: **symbolic-first with an
autograder LLM fallback**. The HuggingFace `math-verify` library checks symbolic
equivalence of the model's `\boxed{...}` against `expected_answer`; only on a
symbolic miss is the judge asked "is this answer Correct or Incorrect?". Answers
math-verify already accepts never reach the judge, so grading stays deterministic
for the large majority of rollouts.

The judge is a **dedicated** model (`judge_nemotron3ultra.yaml` →
Nemotron 3 Ultra by default), not the policy model, so the grading standard stays
fixed when comparing different policy models. Point `judge_model` at a different endpoint to swap it.

### Grading vs. the leaderboard

MathArena grades ArXivMath with its own SymPy-based parser and no LLM judge
("Requires judging: No" in [eth-sri/matharena](https://github.com/eth-sri/matharena)).
That parser is not reused here: it pins `antlr4-python3-runtime==4.11`, which is
incompatible with the `==4.9.*` that Hydra/OmegaConf — Gym's config system —
require.

Instead `math-verify` does the symbolic check and the autograder judge covers
what it misses. Replaying MathArena's published model outputs against their own
`correct` labels, symbolic-only agrees on 93-95% of rollouts, and every
disagreement is a false negative (a correct answer marked wrong, e.g.
`(\gamma-3)\cdot s` vs `s(\gamma-3)`) — no false positives. The judge recovers
those, so scores here can run slightly above the public leaderboard.

## Prompt

Byte-aligned with MathArena's own ArXivMath prompt
(`configs/competitions/arxiv/april.yaml`):

```
You are given a difficult question. Your task is to solve the problem.
Put the final answer you find within \boxed{}.

<question>
```

## Data preparation

```bash
gym eval prepare --benchmark arxivmath_0426
```

Writes `data/arxivmath_0426_benchmark.jsonl` with one row per problem:
`{"question": "...", "expected_answer": "..."}`. The HuggingFace dataset
revision is pinned in `prepare.py` (`HF_REVISION`) for reproducibility.

## Running servers

```bash
gym env start \
    --model-type inference_provider \
    --benchmark arxivmath_0426
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent arxivmath_0426_math_with_autograder_simple_agent \
    --input benchmarks/arxivmath_0426/data/arxivmath_0426_benchmark.jsonl \
    --output results/arxivmath_0426_rollouts.jsonl \
    --num-repeats 4
```

The judge needs `NVIDIA_API_KEY` in the environment.
