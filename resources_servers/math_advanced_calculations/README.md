# Description

1. Environment: This is a tool use - multi step agentic environment involving math problems. 
2. Domain: Math
3. Example prompt: Get me the values for sin(2.0), (1.0 / 1.0), (8.0 + 3.0), (2.0 - 5.0), and (8.0 * 0.0).
4. Verification: reward is 1.0 only if both hold, otherwise 0.0: (a) every expected value is exactly equal to the `solution` of some `function_call_output` item in the trajectory; (b) the final assistant message states the expected values, in order, as its last N answers (N = number of expected values), to about three significant figures. The answer of each line, list item or clause is its last number, ignoring text in parentheses, so restated expressions such as `sin(5.0)` and values that are quoted and then replaced do not count.

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
