# IMO GradingBench (resource server)

Four-class grading of math proofs. The policy model reads a problem
statement plus a candidate proof and emits one of the words
`incorrect | partial | almost | correct` as the **last word** of its
response. Verification is fully deterministic — a single regex strips
markdown / punctuation from the last token, lowercases it, and matches
against the four-grade vocabulary — so **no external LLM judge is
called**. This mirrors NeMo Skills' `gradingbench` metric type
(`GradingBenchMetrics._extract_grade`).

The primary consumer is the [`imo_gradingbench`](../../benchmarks/imo_gradingbench)
benchmark.

## Metrics

The `aggregate_metrics` endpoint emits Gym's standard pass@k /
pass@1[avg-of-k] / majority@k for:

* `exact_accuracy` — 1.0 if the parsed grade equals gold, else 0.0
* `binarized_accuracy` — 1.0 if pred and gold fall in the same bucket
  (high = `{correct, almost}`, low = `{partial, incorrect}`), else 0.0
* `no_answer` — fraction of rollouts whose last word didn't match a
  valid grade

Plus a Skills-parity Mean Absolute Error over the ordinal
`GRADE_TO_SCORE = {correct: 7, almost: 6, partial: 1, incorrect: 0}`
mapping:

* `pass@1[avg-of-k]/mae` — average `|pred_score - gold_score|` across
  all valid (pred, gold) pairs in the first k rollouts of every task
* `pass@k/mae` — best-of-k: the smallest `score_diff` per task,
  averaged
* `mae` / `mae_count` — Skills' all-rollouts pooled value, kept for
  backward-compatible read-out

## JSONL schema

Each input row sets `expected_answer` (the gold grade word) alongside
the baked-in prompt messages. The verifier reads this directly from
the request body and never re-parses the prompt.

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "<grading prompt>"}]},
  "expected_answer": "correct",
  "grading_id": "...",
  "problem_id": "..."
}
```

## Reasoning-model requirement

For reasoning models that emit `<think>…</think>` CoT (Nemotron,
DeepSeek-R1, Qwen3, …), **enable vLLM's `--reasoning-parser`** so the
CoT is routed to a separate reasoning output item and doesn't reach
the last-word extractor:

```bash
vllm serve <model> --reasoning-parser deepseek_r1   # for <think>-style models
```

Without the parser, the "last word" of the model output is the last
word of the *reasoning trace*, which is almost never the committed
grade.

## Example usage

```bash
# Running servers
gym env start \
    --model-type vllm_model \
    --resources-server imo_gradingbench

# Collecting rollouts (5-example smoke test)
gym eval run --no-serve \
    --agent imo_gradingbench_simple_agent \
    --input resources_servers/imo_gradingbench/data/example.jsonl \
    --output results/imo_gradingbench_rollouts.jsonl \
    --num-repeats 1
```
