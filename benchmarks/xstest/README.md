# Prerequisites
Running this benchmark requires 1 GPU for 7B https://huggingface.co/allenai/wildguard

# Prepare data
```bash
gym eval prepare --benchmark xstest
```

# Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark xstest \
    --output results/benchmarks/xstest.jsonl \
    --split benchmark \
    --model-url <> \
    --model-api-key <> \
    --model <> \
    --resume \
    ++overwrite_metrics_conflicts=true \
    ++reuse_existing_data_preparation=true
```
