# MMLU-Redux

[MMLU-Redux 2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux) is a cleaned and corrected variant of MMLU (four-way MCQ). Rows with unresolved errors in the source dataset are skipped, matching NeMo-Skills behavior.

## Usage

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mmlu-redux/config.yaml]"

ng_run "+config_paths=[benchmarks/mmlu-redux/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  "+config_paths=[benchmarks/mmlu-redux/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +output_jsonl_fpath=results/mmlu-redux.jsonl
```
