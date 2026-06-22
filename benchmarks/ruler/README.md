# Prerequisites
Please ensure that you have git-lfs installed!

Linux: `apt update && apt install -y git-lfs`

# Prepare data
```bash
config_paths="benchmarks/ruler/config_nemotron_3_256k.yaml"
ng_prepare_benchmark "+config_paths=[$config_paths]"
```

# Run
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/ruler/config_nemotron_3_256k.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/benchmarks/ruler.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```
