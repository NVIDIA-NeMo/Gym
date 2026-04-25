# MMMU-Pro

[MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) is a multimodal multiple-choice benchmark (up to 10 options A–J) with a vision-critical ``vision`` configuration.

This Gym port follows NeMo-Skills ``mmmu-pro`` preparation: images are saved under ``benchmarks/mmmu-pro/data/media/``, and each JSONL row carries pre-built ``responses_create_params`` plus ``verifier_metadata.media_dir`` for image injection.

## Configuration

- **Agent**: ``responses_api_agents/labbench2_vlm_agent/app.py`` (same embedding pipeline as LabBench2 VLM)
- **Verifier**: ``mcqa`` with ``lenient_answer_colon_md`` (``Answer: X`` style)
- **``media_base_dir``**: ``benchmarks/mmmu-pro/data``

## Usage

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mmmu-pro/config.yaml]"

ng_run "+config_paths=[benchmarks/mmmu-pro/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  "+config_paths=[benchmarks/mmmu-pro/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +output_jsonl_fpath=results/mmmu-pro.jsonl
```

HuggingFace access may require a token in your Gym global config (same as other gated datasets).
