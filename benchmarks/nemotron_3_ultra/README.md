# Pre-requisites
Hardware required for remote config (total 1 GPU):
- xstest: 1 GPU for 7B https://huggingface.co/allenai/wildguard

Hardware required for local config (total 25 GPUs):
1. Ultra model: default configuration is dp4pp1tp4, which requires 16 GPUs. This was primarily tested on GB200 where each node is 4 GPUs, resulting in 4 nodes.
2. xstest: 1 GPU for 7B https://huggingface.co/allenai/wildguard
3. browsecomp: 8 GPUs for dp2pp1tp4 Qwen3-235B-A22B-Instruct-2507-FP8

For Ruler benchmark, please ensure that you have git-lfs installed!

Linux: `apt update && apt install -y git-lfs`

# Configuration
Gated HuggingFace datasets and models to request access to
- xstest: https://huggingface.co/allenai/wildguard
- GPQA: https://huggingface.co/datasets/Idavidrein/gpqa

Please put the following secrets into the local env.yaml
```yaml
wandb_api_key: ???  # For uploading benchmark logs and results
hf_token: ???  # For gated datasets like GPQA

tavily_search_resources_server:
  resources_servers:
    tavily_search:
      tavily_api_key: ???
      exclude_domains_file_path: ???
Qwen3-235B-A22B-Instruct-2507-FP8:
  responses_api_models:
    vllm_model:
      model: ???  # The actual judge model needs to be Qwen/Qwen3-235B-A22B-Instruct-2507-FP8, but the name will probably differ based on endpoint.
      base_url: ???
      api_key: ???
```

# Prepare benchmark data
```bash
config_paths="benchmarks/nemotron_3_ultra/config_short.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```

# Run
## Against an external endpoint
```bash
WANDB_PROJECT=<>
EXPERIMENT_NAME=<>

config_paths="benchmarks/nemotron_3_ultra/remote_endpoint.yaml,\
benchmarks/nemotron_3_ultra/config_short.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```

## Using local compute
```bash
WANDB_PROJECT=<>
EXPERIMENT_NAME=<>

config_paths="benchmarks/nemotron_3_ultra/local_endpoint.yaml,\
benchmarks/nemotron_3_ultra/config_short.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++reuse_existing_data_preparation=true
```


# Configs
We provide two configs: short and long. Short configs are meant to be run on every checkpoint while long configs are meant to be run on every major checkpoint. The benchmarks in the long config are typically more expensive cost-wise to run. For example, Browsecomp uses Tavily API keys for search, which may end up with hundreds of dollars spent per benchmark run.
