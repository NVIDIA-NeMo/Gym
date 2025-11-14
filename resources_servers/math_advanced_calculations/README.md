# Description

1. Environment: This is a tool use - multi step agentic environment involving math problems. 
2. Domain: Math
3. Example prompt: Get me the values for sin(2.0), (1.0 / 1.0), (8.0 + 3.0), (2.0 - 5.0), and (8.0 * 0.0).

Commands - 
Spin up server:

```
  config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_advanced_calculations/configs/math_advanced_calculations.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```
ng_collect_rollouts +agent_name=math_advanced_calculations_simple_agent \
    +input_jsonl_fpath=resources_servers/math_advanced_calculations/data/train.jsonl \
    +output_jsonl_fpath=results/math_advanced_calculations_trajectory_collection.jso
nl \
   +limit=1
```

Data links: https://huggingface.co/datasets/nvidia/Nemotron-RL-math-advanced_calculations 

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
