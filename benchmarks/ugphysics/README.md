# UGPhysics

Undergraduate-physics benchmark from
[YangLabHKUST/UGPhysics](https://github.com/YangLabHKUST/UGPhysics)
([HuggingFace](https://huggingface.co/datasets/UGPhysics/ugphysics)).
~5k free-form physics problems spanning 13 subjects (atomic physics,
classical electromagnetism, classical mechanics, electrodynamics,
geometrical optics, quantum mechanics, relativity, semiconductor
physics, solid-state physics, statistical mechanics, theoretical
mechanics, thermodynamics, wave optics).

Each row carries a problem statement, a canonical reference solution,
the expected boxed answer, an answer-type tag (numerical / expression /
multiple choice / true-or-false / interval / equation / tuple), and a
flag for multi-answer problems.  The benchmark prompt instructs the
model to respond with `\boxed{answer}(unit)` (or
`\boxed{multiple answers connected with commas}` for multi-answer
problems).

## Verification

`ugphysics_judge` symbolic-first cascade:

1. `math_verify` symbolic equivalence of the boxed answer.  Match →
   `TRUE`, judge skipped.
2. On a symbolic miss, an LLM judge is asked once
   ("`## Equivalence Judgement\nTRUE|FALSE`") whether the student's
   answer matches the reference, given the problem, reference solution,
   and reference answer.

The judge prompt is char-for-char Skills'
`nemo_skills/prompt/config/judge/ugphysics.yaml` (four physics
few-shots, ending in `=== report over ===`), and the verdict parser
mirrors `UGPhysicsMetrics.is_correct_judgement` exactly.

## Default judge

`openai/gpt-oss-20b` via NVIDIA's public NIM API
(`integrate.api.nvidia.com`); set `NVIDIA_API_KEY` in your shell.
Skills' baseline judge is `o4-mini-2025-04-16` via OpenAI — for
apples-to-apples migration parity, see the recipe scripts in
`migrate-gym-ugphysics/`.

## Metrics

- **Tier 1** — pass@k / pass@1[avg-of-k] / majority@k for both
  `judge_accuracy` and `symbolic_accuracy`.
- **Tier 2** — per-subject pass@k via `compute_subset_metrics(subset_key="subject")`.
  Mirrors Skills' `subset_for_metrics` stratification across the 13
  UGPhysics subjects.

## Reasoning models

When the policy model emits `<think>…</think>`, start vLLM with
`--reasoning-parser <name>` (e.g. `deepseek_r1` for Nemotron-3-Nano)
so chain-of-thought is routed to a separate reasoning output item and
the `\boxed{…}` extractor / judge prompt only see the final assistant
message.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark ugphysics

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark ugphysics

# Collecting rollouts
gym eval run --no-serve \
    --agent ugphysics_ugphysics_judge_simple_agent \
    --input benchmarks/ugphysics/data/ugphysics_benchmark.jsonl \
    --output results/ugphysics_rollouts.jsonl \
    --num-repeats 4 \
    --max-output-tokens 16384 \
    --temperature 1.0 \
    --top-p 0.95 \
    --prompt-config benchmarks/ugphysics/prompts/default.yaml
```
