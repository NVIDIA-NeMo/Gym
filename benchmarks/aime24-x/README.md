# AIME24-X

Multilingual AIME 2024 benchmark ported from NeMo Skills'
`nemo_skills/dataset/aime24-x`.

## What Is Different From `aime24`

- Source dataset: `nvidia/Nemotron-Multilinugual-Eval-AIME24`
- Languages: `de`, `es`, `fr`, `ja`
- Each row preserves:
  - `subset_for_metrics`: language code
  - `target_language`: language code
- Prompting mirrors Skills' `generic/default` behavior: the full instruction is
  baked into each row's `question`, and the prompt template is a passthrough.

## Verification

This benchmark reuses `math_with_judge` in symbolic-only mode
(`should_use_judge: false`) to match Skills' `++eval_type=math` default.

## Data Preparation

```bash
gym eval prepare --benchmark aime24-x
```

That writes `benchmarks/aime24-x/data/aime24-x_benchmark.jsonl`.

If you want English instructions instead of target-language instructions in the
prepared `question` field, run the script directly:

```bash
python benchmarks/aime24-x/prepare.py --prompt_language en
```

## Quickstart

```bash
gym env start \
    --benchmark aime24-x \
    --model-type vllm_model
```

Then in another shell:

```bash
gym eval run --no-serve \
    --agent aime24-x_math_with_judge_simple_agent \
    --input benchmarks/aime24-x/data/aime24-x_benchmark.jsonl \
    --output results/aime24-x/rollouts.jsonl \
    --num-repeats 32 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-output-tokens 65536 \
    +num_repeats_add_seed=true
```
