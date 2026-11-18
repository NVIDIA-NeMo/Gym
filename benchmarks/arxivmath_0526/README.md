# ArXivMath 05/2026

Research-level final-answer math problems from
[MathArena](https://matharena.ai/arxivmath/)'s ArXivMath **05/2026** release,
sourced from `MathArena/arxivmath-0526` on HuggingFace (40 problems). Problems
are drawn from arXiv papers published that month; answers are a single numeric
value or a pure LaTeX expression.

MathArena publishes a new ArXivMath problem set each month and scores each
release on its own leaderboard, so every month is a separate Gym benchmark
(see also `arxivmath_0426`). Earlier releases (12/2025–03/2026) are marked
deprecated upstream due to contamination risk and are not mirrored here.

## Verification

Uses `math_with_judge`: **symbolic-first with a dedicated Luna medium fallback**.
The final response must contain a nonempty, complete `\boxed{...}`. Missing or
empty boxes receive zero without a judge call. `math-verify` checks symbolic
equivalence first; only misses reach Luna with the raw boxed answer, reference,
and question. A positive judgment is checked again with the answers swapped,
and both judgments must be positive for credit.

The judge is separate from the policy model.

MathArena grades final-answer math symbolically without an LLM fallback.
Gym's parser and fallback differ, so its scores are not an exact reproduction
of the leaderboard's grading methodology.

## Prompt

Byte-aligned with MathArena's own ArXivMath prompt
(`configs/competitions/arxiv/may.yaml`):

```
You are given a difficult question. Your task is to solve the problem.
Put the final answer you find within \boxed{}.

<question>
```

## Data preparation

```bash
gym eval prepare --benchmark arxivmath_0526
```

Writes `data/arxivmath_0526_benchmark.jsonl` with one row per problem:
`{"question": "...", "expected_answer": "..."}`. The HuggingFace dataset
revision is pinned in `prepare.py` (`HF_REVISION`) for reproducibility.

## Running servers

```bash
gym env start \
    --model-type inference_provider \
    --benchmark arxivmath_0526
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent arxivmath_0526_math_with_judge_simple_agent \
    --input benchmarks/arxivmath_0526/data/arxivmath_0526_benchmark.jsonl \
    --output results/arxivmath_0526_rollouts.jsonl \
    --num-repeats 16
```

The judge needs `OPENAI_API_KEY` (or `JUDGE_API_KEY`) in the environment.
The shared [judge config](../judge_luna.yaml) uses the public OpenAI Responses
API. For another compatible provider, set `JUDGE_BASE_URL`, `JUDGE_MODEL`, and
`JUDGE_API_KEY` together. It must support medium reasoning through the Responses
API. The example supplies all repeats at collection time; do not also repeat
the prepared dataset.
