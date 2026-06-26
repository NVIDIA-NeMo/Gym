# Run instructions
1. Update your env.yaml file with the necessary keys and configs. You can leave `exclude_domains_file_path` as a path to an empty json.
```yaml
browsecomp_benchmark_resources_server:
  resources_servers:
    browsecomp_advanced_harness:
      tavily_api_key: ???
      exclude_domains_file_path: ???
Qwen3-235B-A22B-Instruct-2507-FP8:
  responses_api_models:
    vllm_model:
      base_url: ???
      api_key: ???
```

2. Prepare the benchmark dataset
```bash
gym eval prepare --benchmark browsecomp
```

3. Run the benchmark against a VLLMModel
```bash
WANDB_PROJECT=
EXPERIMENT_NAME=
gym eval run \
    --model-type vllm_model \
    --benchmark browsecomp \
    --output results/$EXPERIMENT_NAME.jsonl \
    --split benchmark \
    --model-url ??? \
    --model-api-key ??? \
    --model ??? \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME
```
