# Prepare data
```bash
config_paths="nemo_gym/benchmarks/aalcr/config.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```

# Run
```bash
config_paths="nemo_gym/responses_api_models/vllm_model/configs/vllm_model.yaml,\
nemo_gym/benchmarks/aalcr/config.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/nemo_gym/benchmarks/aalcr.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++ray_head_node_address=auto \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<> \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.base_url=<>' \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.model=<>' \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.api_key=<>'
```
