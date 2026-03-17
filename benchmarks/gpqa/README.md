# GPQA Diamond

[GPQA](https://arxiv.org/abs/2311.12022) (Graduate-Level Google-Proof Q&A) Diamond is a challenging multiple-choice question answering benchmark with graduate-level questions across physics, biology, and chemistry.

## Configuration

This benchmark uses the `mcqa` resource server with the `mcqa_simple_agent`.

- **num_repeats**: 8
- **Grading mode**: `lenient_answer_colon_md` (markdown-aware `Answer: X` extraction, matching NeMo-Skills evaluator behavior)
- **Prompt**: `Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' ...`

## Usage

```bash
# Prepare data
ng_prepare_benchmark "+config_paths=[benchmarks/gpqa/config.yaml]"

# Start servers
ng_run "+config_paths=[benchmarks/gpqa/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts
ng_collect_rollouts \
    "+config_paths=[benchmarks/gpqa/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_jsonl_fpath=results/gpqa.jsonl
```
