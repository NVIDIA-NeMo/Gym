# AIME25-X

Multilingual AIME 2025 benchmark ported from NeMo Skills'
`nemo_skills/dataset/aime25-x`.

## What Is Different From `aime25`

- Source dataset: `nvidia/Nemotron-Multilinugual-Eval-AIME25`
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
ng_prepare_benchmark "+config_paths=[benchmarks/aime25-x/config.yaml]"
```

That writes `benchmarks/aime25-x/data/aime25-x_benchmark.jsonl`.

If you want English instructions instead of target-language instructions in the
prepared `question` field, run the script directly:

```bash
python benchmarks/aime25-x/prepare.py --prompt_language en
```

## Quickstart

```bash
ng_run "+config_paths=[benchmarks/aime25-x/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Then in another shell:

```bash
ng_collect_rollouts \
    +agent_name=aime25-x_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/aime25-x/data/aime25-x_benchmark.jsonl \
    +output_jsonl_fpath=results/aime25-x/rollouts.jsonl \
    +prompt_config=benchmarks/prompts/generic_default.yaml \
    +num_repeats=32 +num_repeats_add_seed=true \
    "+responses_create_params={temperature: 1.0, top_p: 0.95, max_output_tokens: 65536}"
```
