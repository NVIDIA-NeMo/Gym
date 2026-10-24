# APEX 2025

Math problems from MathArena's APEX 2025 benchmark, sourced from
`MathArena/apex_2025` on HuggingFace. This benchmark is intended as a
newer alternative to `apex_shortlist`.

## Verification

Reuses the `math_with_judge` resource server in **symbolic-only** mode
(`should_use_judge: false`) to mirror NeMo Skills' `eval_type=math`
default for this benchmark. The HuggingFace `math-verify` library does
symbolic equivalence of the model-extracted `\boxed{...}` answer against
`expected_answer`.

## Prompt

User-only prompt, character-for-character match with NeMo Skills'
`generic/math.yaml`:

```
Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

<question>
```

## Data preparation

```bash
ng_prepare_benchmark '+config_paths=[benchmarks/apex_2025/config.yaml]'
```

Writes `data/apex_2025_benchmark.jsonl` with one row per problem:
`{"problem_idx": 1, "source": "...", "question": "...", "expected_answer": "..."}`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/apex_2025/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=apex_2025_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/apex_2025/data/apex_2025_benchmark.jsonl \
    +output_jsonl_fpath=results/apex_2025_rollouts.jsonl \
    +num_repeats=4
```
