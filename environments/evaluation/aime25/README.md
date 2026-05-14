# Description

AIME 2025 math benchmark. 30 competition problems requiring exact integer answers.

1. Domain: Math
2. Source: https://huggingface.co/datasets/MathArena/aime_2025

Commands -
Prepare data:

```
python environments/evaluation/aime25/prepare.py
```

Spin up server:

```
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
environments/evaluation/aime25/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```
ng_collect_rollouts +agent_name=aime25_math_with_judge_simple_agent \
    +input_jsonl_fpath=environments/evaluation/aime25/data/aime25_benchmark.jsonl \
    +output_jsonl_fpath=results/aime25_rollouts.jsonl \
    +num_repeats=32
```

# Licensing information
Code: Apache 2.0
Data: MIT
