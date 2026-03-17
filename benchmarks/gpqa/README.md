# GPQA Diamond

[GPQA](https://arxiv.org/abs/2311.12022) (Graduate-Level Google-Proof Q&A) Diamond is a challenging multiple-choice question answering benchmark with graduate-level questions across physics, biology, and chemistry.

## Configuration

This benchmark uses the `mcqa` resource server with the `mcqa_simple_agent`.

- **num_repeats**: 8
- **Sampling**: temperature=1.0, max_output_tokens=65536
- **Grading mode**: strict_single_letter_boxed

## Usage

```bash
# Prepare data
python benchmarks/gpqa/prepare.py

# Start servers
ng_run "+config_paths=[benchmarks/gpqa/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts
ng_collect_rollouts \
    "+config_paths=[benchmarks/gpqa/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_jsonl_fpath=results/gpqa.jsonl
```
