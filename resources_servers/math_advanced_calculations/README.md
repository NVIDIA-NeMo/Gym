# Description

1. Environment: This is a tool use - multi step agentic environment involving math problems. 
2. Domain: Math
3. Source of prompts: 
- Hard set of prompts (100): https://huggingface.co/datasets/Nexusflow/MultiverseMathHard 
- Full set of prompts (6K): https://huggingface.co/datasets/Nexusflow/multiverse_math_reflection_benchmark
4. Example prompt: Get me the values for sin(2.0), (1.0 / 1.0), (8.0 + 3.0), (2.0 - 5.0), and (8.0 * 0.0).
5. Verifier: The verifier accuracy is checked by running the same benchmark on Nexusbench and making the sure the average reward of trajectories collected is within variance of the Nexusbench accuracy.
6. Legal Approval Status: TBD

Rollouts - 
Link: https://huggingface.co/datasets/Nexusflow/abhibha-rollouts-mmhs

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

Data links: https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/46/versions/55#/

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
