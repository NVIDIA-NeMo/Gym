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
ng_prepare_benchmark "+config_paths=[benchmarks/gpqa_x/config.yaml]"
```

That writes `benchmarks/gpqa_x/data/gpqa_x_benchmark.jsonl`.

If you want English instructions instead of target-language instructions in the
prepared `question` field, run the script directly:

```bash
python benchmarks/gpqa_x/prepare.py --prompt_language en
```

## Quickstart

```bash
ng_run "+config_paths=[benchmarks/gpqa_x/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Then in another shell:

```bash
ng_collect_rollouts \
    +agent_name=gpqa_x_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/gpqa_x/data/gpqa_x_benchmark.jsonl \
    +output_jsonl_fpath=results/gpqa_x/rollouts.jsonl \
    +prompt_config=benchmarks/gpqa_x/prompts/default.yaml \
    +num_repeats=8 +num_repeats_add_seed=true \
    "+responses_create_params={temperature: 1.0, top_p: 0.95, max_output_tokens: 32768}"
```
