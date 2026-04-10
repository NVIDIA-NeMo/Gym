# Prepare data
```bash
config_paths="benchmarks/spider2_lite/config.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```

# Run
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/spider2_lite/config.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/benchmarks/spider2_lite.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++spider2_lite_benchmark_resources_server.resources_servers.spider2_lite.max_concurrency=8 \
    ++resume_from_cache=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```
