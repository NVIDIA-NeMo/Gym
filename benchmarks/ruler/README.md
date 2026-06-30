# Prerequisites
Please ensure that you have git-lfs installed!

Linux: `apt update && apt install -y git-lfs`

# Prepare data
```bash
gym eval prepare --benchmark ruler/config_nemotron_3_256k
```

# Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark ruler/config_nemotron_3_256k \
    --output results/benchmarks/ruler.jsonl \
    --split benchmark \
    --model-url <> \
    --model-api-key <> \
    --model <> \
    --resume \
    ++overwrite_metrics_conflicts=true \
    ++reuse_existing_data_preparation=true
```
