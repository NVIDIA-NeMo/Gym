# IMO AnswerBench

Math benchmark from [google-deepmind/superhuman](https://github.com/google-deepmind/superhuman/) — short-answer IMO-style problems. Ported from NeMo Skills' `imo-answerbench`.

## Prepare data

`prepare.py` downloads `answerbench_v2.csv` from the exact pinned superhuman commit that Skills uses, and writes `data/imo_answerbench_benchmark.jsonl` with one row per problem (`question`, `expected_answer`, plus `problem_id` / `category` / `subcategory` / `source`).

```bash
gym eval prepare --benchmark imo_answerbench
```

## Run servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark imo_answerbench
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent imo_answerbench_math_with_autograder_simple_agent \
    --input benchmarks/imo_answerbench/data/imo_answerbench_benchmark.jsonl \
    --output results/imo_answerbench_rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/imo_answerbench/prompts/default.yaml \
    +num_repeats_add_seed=true
```

## Verification

Two-stage, matching Skills:

1. **Symbolic check** via `math-verify` on the `\boxed{...}` answer (inherited from the `math_with_judge` resource server).
2. **LLM-autograder fallback** when symbolic fails. The benchmark binds to the `math_with_autograder` resource server, which subclasses `math_with_judge` to swap in a Skills-style autograder judge. The autograder prompt is the same one NeMo Skills uses (`nemo_skills/prompt/config/judge/imo_answerbench.yaml`) — it asks the judge whether the model's answer is `\boxed{Correct}` or `\boxed{Incorrect}` against the expected answer.

> **Reasoning-model note**: start the policy vLLM server with `--reasoning-parser deepseek_r1` (or the model-specific parser). That strips `<think>…</think>` at the model edge, so `\boxed{...}` extraction and the judge both see clean post-think text. Without it the judge prompt is polluted with chain-of-thought and symbolic extraction can miss the final boxed answer.

The default judge endpoint is `openai/gpt-oss-20b` via NVIDIA's public NIM API (`integrate.api.nvidia.com`, authed with `NVIDIA_API_KEY`). See `judge_gptoss20b.yaml`. To swap to another OpenAI-compatible endpoint, override `judge_base_url` / `judge_api_key` / `judge_model_name`.

## Metrics

The `math_with_autograder` resource server inherits its metric set from `math_with_judge` (via `compute_pass_majority_metrics`):

- `pass@1[avg-of-k]/symbolic_accuracy`, `pass@k/symbolic_accuracy` (symbolic only)
- `pass@1[avg-of-k]/judge_accuracy`, `pass@k/judge_accuracy` (judge-pass rate on rollouts that fell through to the judge)
- `majority@k/...` (requires `answer_key="extracted_answer"`, already set)
