# BixBench (Aviary)

Adapts the BixBench [Aviary environment](https://github.com/Future-House/aviary) into NeMo Gym. Implements the [BixBench dataset](https://arxiv.org/abs/2503.00096) an environment with execution of a Jupyter notebook. Also serves as an example for how to implement notebook-backed environments for other scientific computational tasks.

# Example usage

```bash
config_paths="environments/aviary_bixbench/config.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

```bash
ng_collect_rollouts \
    +agent_name=bixbench_aviary_agent +input_jsonl_fpath=environments/aviary_bixbench/data/example.jsonl \
    +output_jsonl_fpath=environments/aviary_bixbench/data/example_rollouts.jsonl
```

# Licensing information

Code: Apache 2.0
