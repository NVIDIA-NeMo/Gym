# Prepare data
```bash
gym eval prepare --benchmark spider2_lite
```

# Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark spider2_lite \
    --output results/benchmarks/spider2_lite.jsonl \
    --split benchmark \
    --model-url <> \
    --model-api-key <> \
    --model <> \
    --resume \
    ++overwrite_metrics_conflicts=true \
    ++spider2_lite_benchmark_resources_server.resources_servers.spider2_lite.max_concurrency=8 \
    ++reuse_existing_data_preparation=true
```
