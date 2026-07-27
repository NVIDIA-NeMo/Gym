# Prepare data
```bash
gym eval prepare --benchmark aalcr
```

# Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark aalcr \
    ++output_jsonl_fpath=results/benchmarks/aalcr.jsonl \
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
