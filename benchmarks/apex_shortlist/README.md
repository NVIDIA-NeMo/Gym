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
gym eval prepare --benchmark apex_shortlist
```

Writes `data/apex_shortlist_benchmark.jsonl` with one row per problem:
`{"question": "...", "expected_answer": "..."}`.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark apex_shortlist
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent apex_shortlist_math_with_judge_simple_agent \
    --input benchmarks/apex_shortlist/data/apex_shortlist_benchmark.jsonl \
    --output results/apex_shortlist_rollouts.jsonl \
    --num-repeats 4
```
