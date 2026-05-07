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

References the shared `benchmarks/prompts/generic_math.yaml` — the same
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
ng_prepare_benchmark "+config_paths=[benchmarks/hmmt_nov25/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/hmmt_nov25/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=hmmt_nov25_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/hmmt_nov25/data/hmmt_nov25_benchmark.jsonl \
    +output_jsonl_fpath=results/hmmt_nov25_rollouts.jsonl \
    +prompt_config=benchmarks/prompts/generic_math.yaml \
    +num_repeats=16 \
    +num_repeats_add_seed=true \
    "+responses_create_params={temperature: 1.0, top_p: 0.95, max_output_tokens: 65536}"
```
