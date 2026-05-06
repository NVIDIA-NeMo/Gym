# MMLU-ProX

[MMLU-ProX](https://arxiv.org/abs/2503.04861) is a multilingual extension of MMLU-Pro with 10 answer choices (A–J) across 6 languages: English, German, Spanish, French, Italian, and Japanese. Questions are professionally translated and include language-specific answer extraction patterns.

## Configuration

This benchmark uses the `mcqa` resource server with the `mcqa_simple_agent`.

- **Grading mode**: `null` — each row supplies its own language-specific extraction regex via `template_metadata.output_regex`
- **Prompt**: Passthrough (`{question}` only) — the complete formatted question including options is baked into the data during preparation

## Usage

```bash
# Prepare data
ng_prepare_benchmark "+config_paths=[benchmarks/mmlu_prox/config.yaml]"

# Start servers
ng_run "+config_paths=[benchmarks/mmlu_prox/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts
ng_collect_rollouts \
    "+config_paths=[benchmarks/mmlu_prox/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_jsonl_fpath=results/mmlu_prox.jsonl
```
