# MMLU

[Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) is a 57-subject multiple-choice benchmark (four options A–D).

This Gym benchmark ports NeMo-Skills' `mmlu` data preparation: per-subject test splits are loaded from HuggingFace (`lukaemon/mmlu`, falling back to `cais/mmlu`), then converted to `mcqa` JSONL with `lenient_answer_colon_md` grading (aligned with `benchmarks/mmlu_pro`).

## Usage

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mmlu/config.yaml]"

ng_run "+config_paths=[benchmarks/mmlu/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  "+config_paths=[benchmarks/mmlu/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +output_jsonl_fpath=results/mmlu.jsonl
```
