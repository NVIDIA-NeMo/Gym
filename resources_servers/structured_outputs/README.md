# Description
> Keywords: Instruction Following, Structured Outputs, Schema Adherence

This is a resource server for verifying the ability of the model to follow output formatting instructions under schema constraints.

Each problem consists three components:
1. Document
2. Output Formatting Instruction (Schema)
3. Question

The dataset can be found at https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs 


We recommend formatting the dataset to test the model's ability to follow instructions under the following circumstances:
1. Different Instruction Locations
   1. The instruction can be in the system or user message, and can be before or after the question.
2. Difficulty of Instructions
   1. The instruction can be simple, or detailed
      1. e.g. simple: `Schema: {schema}`
      2. e.g. detailed `Please format your answer using the following schema: {schema}. Remember to validate all typing and formatting constraints. Do not format your answer in Markdown,`
3. Difficulty of Question
   1. The question exists only to serve as a proxy for eliciting a response worthy of output formatting. To focus the environment towards schema adherence, the question should be easy.
      1. e.g. simple: `Please provide a response based on the document and provided schema`.

For the JSON variant, we use the `openapi-schema-validator` library for verification.

> [!IMPORTANT]
> Evaluation is only based on the **schema adherence** of the generated output.
> **The actual content of the generation is *not* verified**, thus it is advised that the task used for prompt creation is not too difficult for the model.


# Example usage

## Running servers
The following are example commands for running this resource server, along with the simple agent and an OpenAI model:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/structured_outputs/configs/structured_outputs_json.yaml"
ng_run "+config_paths=[$config_paths]"
```

The dataset can be found at https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs 

Then, rollouts can be collected using a command such as the following:
```bash
ng_collect_rollouts \
    +agent_name=structured_outputs_simple_agent \
    +input_jsonl_fpath=resources_servers/structured_outputs/data/structured_outputs_251027_nano_v3_sdg_json_train.jsonl \
    +output_jsonl_fpath=results/example_structured_outputs_json.jsonl \
    +limit=1
```

You can prepare the data for training with:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_json.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/structured_outputs \
    +mode=train_preparation +should_download=true
```

# Licensing information
Code: Apache 2.0

Data: CC BY 4.0

Dependencies
- nemo_gym: Apache 2.0
- openapi-schema-validator: [BSD-3-Clause license](https://github.com/python-openapi/openapi-schema-validator/blob/master/LICENSE)
