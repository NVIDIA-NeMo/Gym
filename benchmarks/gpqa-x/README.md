# GPQA-X

Multilingual GPQA benchmark ported from NeMo Skills'
`nemo_skills/dataset/gpqa-x`.

## What Is Different From `gpqa`

- Source dataset: `nvidia/Nemotron-Multilinugual-Eval-GPQA`
- Languages: `de`, `es`, `fr`, `ja`
- Multiple-choice question answering with 4 options
- Prompting mirrors Skills' `generic/default` behavior: the full instruction,
  question, and options are baked into each row's `question`, and the prompt
  template is a passthrough.
- Each row carries `template_metadata.output_regex` so the `mcqa` verifier can
  extract the boxed answer letter.

## Verification

This benchmark reuses the `mcqa` resource server, which matches Skills'
`++eval_type=multichoice` default.

## Data Preparation

```bash
gym eval prepare --benchmark gpqa-x
```

That writes `benchmarks/gpqa-x/data/gpqa-x_benchmark.jsonl`.

If you want English instructions instead of target-language instructions in the
prepared `question` field, run the script directly:

```bash
python benchmarks/gpqa-x/prepare.py --prompt_language en
```

## Quickstart

```bash
gym env start \
    --benchmark gpqa-x \
    --model-type vllm_model
```

Then in another shell:

```bash
gym eval run --no-serve \
    --agent gpqa-x_mcqa_simple_agent \
    --input benchmarks/gpqa-x/data/gpqa-x_benchmark.jsonl \
    --output results/gpqa-x/rollouts.jsonl \
    --prompt-config benchmarks/prompts/generic/default.yaml \
    --num-repeats 8 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-output-tokens 32768 \
    +num_repeats_add_seed=true
```
