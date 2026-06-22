# MMLU-Pro

[MMLU-Pro](https://arxiv.org/abs/2406.01574) is a challenging multiple-choice question answering benchmark with 10 answer choices (A–J) across 14 disciplines including math, science, law, business, and more. It extends the original MMLU benchmark with harder questions and more distractor options.

## Configuration

This benchmark uses the `mcqa` resource server with the `mcqa_simple_agent`.

- **Grading mode**: `lenient_answer_colon_md` (markdown-aware `Answer: X` extraction, matching NeMo-Skills evaluator behavior)
- **Prompt**: `Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: $LETTER' where LETTER is one of A, B, C, D, E, F, G, H, I, J. ...`

## Usage

```bash
# Prepare data
ng_prepare_benchmark "+config_paths=[benchmarks/mmlu_pro/config.yaml]"

# Start servers
ng_run "+config_paths=[benchmarks/mmlu_pro/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts
ng_collect_rollouts \
    "+config_paths=[benchmarks/mmlu_pro/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_jsonl_fpath=results/mmlu_pro.jsonl
```
