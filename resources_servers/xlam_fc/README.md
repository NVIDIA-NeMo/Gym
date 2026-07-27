# XlamFc Resources Server

Function calling using the [Salesforce xlam-function-calling-60k dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).


```bash
huggingface-cli login
python resources_servers/xlam_fc/generate_dataset.py
```

```bash
gym env start \
    --model-type vllm_model \
    --resources-server xlam_fc
```

```bash
gym eval run --no-serve \
    --agent xlam_fc_simple_agent \
    --input resources_servers/xlam_fc/data/train.jsonl \
    --output results/xlam_fc_trajectory_collection.jsonl \
    --limit 10
```

## Licensing
Code: Apache 2.0
Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
