# Description

1. Environment: This is a tool use - multi step agentic environment involving math problems. 
2. Domain: Math
3. Example prompt: Get me the values for sin(2.0), (1.0 / 1.0), (8.0 + 3.0), (2.0 - 5.0), and (8.0 * 0.0).

Commands - 
Spin up server:

```
gym env start \
    --model-type openai_model \
    --resources-server math_advanced_calculations
```

Collect trajectories:
```
gym eval run --no-serve \
    --agent math_advanced_calculations_simple_agent \
    --input resources_servers/math_advanced_calculations/data/train.jsonl \
    --output results/math_advanced_calculations_trajectory_collection.jsonl \
   --limit 1
```

Data links: https://huggingface.co/datasets/nvidia/Nemotron-RL-math-advanced_calculations 

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
