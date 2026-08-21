# GSM8k (Aviary)

Adapts the GSM8K [Aviary environment](https://github.com/Future-House/aviary) into NeMo Gym. Implements [GSM8k](https://arxiv.org/abs/2110.14168) as an environment equipped with a calculator tool.

# Example usage

Run the GSM8K Aviary resources server and aviary agent together with a model config:

```bash
config_paths="environments/aviary_gsm8k/config.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

Then collect rollouts:

```bash
ng_collect_rollouts \
    +agent_name=gsm8k_aviary_agent +input_jsonl_fpath=environments/aviary_gsm8k/data/example.jsonl \
    +output_jsonl_fpath=environments/aviary_gsm8k/data/example_rollouts.jsonl
```

# Licensing information

Code: Apache 2.0
Data:
- GSM8K: MIT
