# FrontierScience Olympiad

Free-form science olympiad benchmark from OpenAI's
[FrontierScience](https://huggingface.co/datasets/openai/frontierscience)
release. Three subjects — chemistry, biology, physics — combined into a
single split (~hundreds of problems). Verification is single-pass LLM
judge (`Judgement: YES/NO`).

## Source

- Dataset: `openai/frontierscience`, config `olympiad`, split `test`
- Paper: https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf
- License: Apache 2.0

## Verification

Uses the [`frontierscience_judge`](../../resources_servers/frontierscience_judge/)
resource server. The judge prompt is the verbatim FrontierScience grading
rubric from page 13 of the paper. The default config routes the judge to
`openai/gpt-oss-20b` on the public NVIDIA inference API
(`https://integrate.api.nvidia.com/v1`), reading the key from
`NVIDIA_API_KEY`. Override the top-level `judge_base_url` /
`judge_api_key` / `judge_model_name` vars to swap in another judge — for
example, the original Skills configuration uses `o3-mini-2025-01-31` via
`api.openai.com`.

## Metrics

- `pass@k/accuracy`, `pass@1[avg-of-k]/accuracy`, `majority@k/accuracy`
  — global pass-rates across all three subjects.
- `chemistry/pass@k/accuracy`, `biology/...`, `physics/...` — per-subject
  pass@k via `compute_subset_metrics(subset_key="subject")`. Mirrors
  Skills' `MathMetrics` per-`subset_for_metrics` breakdown.

## Example usage

```bash
# Prepare benchmark data (downloads from HuggingFace)
gym eval prepare --benchmark frontierscience_olympiad

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark frontierscience_olympiad

# Collecting rollouts (4 rollouts per task, matching Skills' default)
gym eval run --no-serve \
    --agent frontierscience_olympiad_frontierscience_judge_simple_agent \
    --input benchmarks/frontierscience_olympiad/data/frontierscience_olympiad_benchmark.jsonl \
    --prompt-config benchmarks/frontierscience_olympiad/prompts/default.yaml \
    --output results/frontierscience_olympiad_rollouts.jsonl \
    --num-repeats 4
```

For Nemotron-3-Nano and other reasoning models, start vLLM with
`--reasoning-parser deepseek_r1` so `<think>...</think>` is stripped at
the model edge before the judge sees it.
