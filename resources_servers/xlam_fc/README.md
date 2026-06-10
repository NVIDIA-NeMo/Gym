# XlamFc Resources Server

Function calling using the [Salesforce xlam-function-calling-60k dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).


```bash
huggingface-cli login
python resources_servers/xlam_fc/generate_dataset.py
```

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/xlam_fc/configs/xlam_fc.yaml"
ng_run "+config_paths=[$config_paths]"
```

```bash
ng_collect_rollouts \
    +agent_name=xlam_fc_simple_agent \
    +input_jsonl_fpath=resources_servers/xlam_fc/data/train.jsonl \
    +output_jsonl_fpath=results/xlam_fc_trajectory_collection.jsonl \
    +limit=10
```

## Licensing
Code: Apache 2.0
Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
