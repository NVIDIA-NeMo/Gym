# HMMT Nov 2025

Problems from the Harvard-MIT Mathematics Tournament (November 2025),
sourced from `MathArena/hmmt_nov_2025` on HuggingFace.

## Verification

Reuses the `math_with_judge` resource server in **symbolic-only** mode
(`should_use_judge: false`) to mirror NeMo Skills' `eval_type=math`
default for this benchmark. The HuggingFace `math-verify` library does
symbolic equivalence of the model-extracted `\boxed{...}` answer against
`expected_answer`. Matches the hmmt_feb25 migration (upstream PR #1112).

## Prompt

References the shared `benchmarks/prompts/generic/math.yaml` — the same
prompt `gsm8k`, `hendrycks_math`, and other `eval_type=math` benchmarks
use. Rendered-equivalent to NeMo Skills' `generic/math.yaml`: Skills'
template is `{examples}{problem}` with `{examples}` empty by default;
the shared Gym prompt collapses that into `{question}`. Both produce
the same user message at rollout time (user-only, no system, no
few-shots).

## Reasoning parser

Start vLLM with the `--reasoning-parser` that matches your model
(e.g. `deepseek_r1` for models with a `<think>…</think>` convention;
the parser name is declared in
`responses_api_models/local_vllm_model/configs/nvidia/*.yaml`). Without
one, `math_with_judge` may extract intermediate expressions from
truncated rollouts, and Skills' `parse_reasoning=True` default diverges
on the same inputs.

## Quickstart

```bash
# Prepare benchmark data (downloads from HuggingFace)
gym eval prepare --benchmark hmmt_nov25

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark hmmt_nov25

# Collecting rollouts
gym eval run --no-serve \
    --agent hmmt_nov25_math_with_judge_simple_agent \
    --input benchmarks/hmmt_nov25/data/hmmt_nov25_benchmark.jsonl \
    --output results/hmmt_nov25_rollouts.jsonl \
    --prompt-config benchmarks/prompts/generic/math.yaml \
    --num-repeats 16 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-output-tokens 65536 \
    +num_repeats_add_seed=true
```
