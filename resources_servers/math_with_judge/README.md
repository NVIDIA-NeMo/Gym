# Description
This is a resources server for verifying the correctness of answers to mathematical problems.  It uses a combination of the Hugging Face Math-Verify library and an LLM as a judge.

The problems in the OpenMathReasoning dataset are taken from the
[OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning)
on Hugging Face.


# Example usage

## Running servers
The following are example commands for running this resources server, along with the simple agent and an OpenAI model:
```bash
gym env start \
    --model-type openai_model \
    --resources-server math_with_judge \
    +math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

To download the OpenMathReasoning dataset, the following command can be run:
```bash
gym dataset download --storage gitlab \
    --name math_open_math_reasoning \
    --revision 0.0.1 \
    --artifact open_math_reasoning_problems.jsonl \
    --output data/open_math_reasoning_problems.jsonl
```

Then, rollouts can be collected using a command such as the following:
```bash
gym eval run --no-serve \
    --agent math_with_judge_simple_agent \
    --input data/open_math_reasoning_problems.jsonl \
    --output results/example_open_math_reasoning_verify_responses.jsonl \
    --limit 5
```

## Prepare for trajectory collection
```bash
gym dataset collate \
    --config resources_servers/math_with_judge/configs/dapo17k_trajectory_collection.yaml \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    --output-dir data/dapo17k_trajectory_collection \
    --mode train_preparation \
    --download
```

# Licensing information
Code: Apache 2.0<br>
Data:
- OpenMathReasoning: Creative Commons Attribution 4.0 International
- Math Stack Overflow: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies
- nemo_gym: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
