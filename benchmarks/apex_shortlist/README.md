# APEX Shortlist

Math problems from MathArena's APEX Shortlist, sourced from
`MathArena/apex-shortlist` on HuggingFace. Mirrors the NeMo Skills
`apex-shortlist` benchmark (`nemo_skills/dataset/apex-shortlist/`).

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
ng_prepare_benchmark '+config_paths=[benchmarks/apex_shortlist/config.yaml]'
```

Writes `data/apex_shortlist_benchmark.jsonl` with one row per problem:
`{"question": "...", "expected_answer": "..."}`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/apex_shortlist/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=apex_shortlist_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/apex_shortlist/data/apex_shortlist_benchmark.jsonl \
    +output_jsonl_fpath=results/apex_shortlist_rollouts.jsonl \
    +num_repeats=4
```
