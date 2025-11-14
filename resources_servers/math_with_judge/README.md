# Description
This is a resource server for verifying the correctness of answers to mathematical problems.  It uses a combination of the Hugging Face Math-Verify library and an LLM as a judge.

The problems in the OpenMathReasoning dataset are taken from the
[OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
on Hugging Face.

The problems in the Math Stack Overflow dataset come from Stack Overflow.

Data links:
- OpenMathReasoning: https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/37#/
- Math Stack Overflow: https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/68#/

# Example usage

## Running servers
The following are example commands for running this resource server, along with the simple agent and an OpenAI model:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/math_with_judge/configs/math_with_judge.yaml"
ng_run "+config_paths=[$config_paths]" \
    +math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

To download the OpenMathReasoning dataset, the following command can be run:
```bash
ng_download_dataset_from_gitlab \
    +dataset_name=math_open_math_reasoning \
    +version=0.0.1 \
    +artifact_fpath=open_math_reasoning_problems.jsonl \
    +output_fpath=data/open_math_reasoning_problems.jsonl
```

Then, rollouts can be collected using a command such as the following:
```bash
ng_collect_rollouts \
    +agent_name=math_with_judge_simple_agent \
    +input_jsonl_fpath=data/open_math_reasoning_problems.jsonl \
    +output_jsonl_fpath=results/example_open_math_reasoning_verify_responses.jsonl \
    +limit=5
```

## Prepare for trajectory collection
```bash
config_paths="resources_servers/math_with_judge/configs/dapo17k_trajectory_collection.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/dapo17k_trajectory_collection \
    +mode=train_preparation \
    +should_download=true
```

# Licensing information
Code: Apache 2.0<br>
Data:
- OpenMathReasoning: Creative Commons Attribution 4.0 International
- Math Stack Overflow: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies
- nemo_gym: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
