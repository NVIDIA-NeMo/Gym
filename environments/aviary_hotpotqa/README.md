# HotPotQA (Aviary)

Adapts the HotPotQA [Aviary environment](https://github.com/Future-House/aviary) into NeMo Gym. The HotPotQA environment asks agents to perform multi-hop question answering on the [HotPotQA dataset](https://aclanthology.org/D18-1259/) with a wikipedia search tool.

# Example usage

```bash
config_paths="environments/aviary_hotpotqa/config.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

```bash
ng_collect_rollouts \
    +agent_name=hotpotqa_aviary_agent +input_jsonl_fpath=environments/aviary_hotpotqa/data/example.jsonl \
    +output_jsonl_fpath=environments/aviary_hotpotqa/data/example_rollouts.jsonl
```

# Licensing information

Code: Apache 2.0
Data:
- HotPotQA: Creative Commons Attribution-ShareAlike 4.0 International
