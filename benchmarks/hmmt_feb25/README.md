# HMMT Feb 2025

30 problems from the Harvard-MIT Mathematics Tournament (February 2025),
sourced from `MathArena/hmmt_feb_2025` on HuggingFace.

## Verification

Reuses the `math_with_judge` resource server in **symbolic-only** mode
(`should_use_judge: false`) to mirror NeMo Skills' `eval_type=math`
default for this benchmark. The HuggingFace `math-verify` library does
symbolic equivalence of the model-extracted `\boxed{...}` answer against
`expected_answer`.

## Prompt

User-only prompt, character-for-character match with NeMo Skills'
`generic/math.yaml` (no few-shots):

```
Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

<question>
```

## Data preparation

```
ng_prepare_benchmark '+config_paths=[benchmarks/hmmt_feb25/config.yaml]'
```

Writes `data/hmmt_feb25_benchmark.jsonl` with one row per problem:
`{"question": "...", "expected_answer": "..."}`.

## Quickstart

Start the benchmark's servers (inherits `math_with_judge` in symbolic-only mode
plus a vLLM model server — adjust the model config to match your deployment):

```
ng_run "+config_paths=[benchmarks/hmmt_feb25/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

In a separate shell, collect rollouts against the full 30-problem set. `num_repeats`
controls rollouts-per-task for pass@k; use 16 for parity-grade evaluation, or drop
to 4 for a faster smoke pass:

```
ng_collect_rollouts \
    +agent_name=hmmt_feb25_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/hmmt_feb25/data/hmmt_feb25_benchmark.jsonl \
    +output_jsonl_fpath=results/hmmt_feb25/rollouts.jsonl \
    +prompt_config=benchmarks/hmmt_feb25/prompts/default.yaml \
    +num_repeats=16 +num_repeats_add_seed=true \
    "+responses_create_params={temperature: 1.0, top_p: 0.95, max_output_tokens: 65536}"
```

`num_repeats_add_seed=true` assigns a distinct vLLM `seed` to each rollout via
`metadata.extra_body`, which the vllm_model server forwards to the sampler.
