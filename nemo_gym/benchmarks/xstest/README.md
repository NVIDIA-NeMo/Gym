# Prerequisites
Running this benchmark requires 1 GPU for 7B https://huggingface.co/allenai/wildguard

# Prepare data
```bash
config_paths="nemo_gym/benchmarks/xstest/config.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```

# Run
```bash
config_paths="nemo_gym/responses_api_models/vllm_model/configs/vllm_model.yaml,\
nemo_gym/benchmarks/xstest/config.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/nemo_gym/benchmarks/xstest.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```
