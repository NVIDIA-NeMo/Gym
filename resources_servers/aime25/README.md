# AIME 2025 Resources Server

Can be used for validation in training.

For testing: 

```
ng_run "+config_paths=[resources_servers/aime25/configs/aime25.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```
ng_collect_rollouts \
    +agent_name=aime25_simple_agent \
    +input_jsonl_fpath=resources_servers/aime25/data/example.jsonl \
    +output_jsonl_fpath=results/aime25_rollouts.jsonl \
    +limit=5
```

Data links: TBD


## Licensing Information

Code: Apache 2.0
Data: TBD

## Dependencies

- nemo_gym: Apache 2.0
- latex2sympy2_extended
- math_verify
