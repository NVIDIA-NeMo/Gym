# Description
> Keywords: Instruction Following, Structured Outputs, Schema Adherence

This is a resources server for verifying the ability of the model to follow output formatting instructions under schema constraints.

Each problem consists three components:
1. Document
2. Output Formatting Instruction (Schema)
3. Question

The dataset can be found at https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs 


We recommend formatting the dataset to test the model's ability to follow instructions under the following circumstances:
1. Different Instruction Locations
   1. The instruction can be in the system or user message, and can be before or after the question.
2. Difficulty of Instructions
   1. The instruction can be simple, or detailed:
      1. e.g. simple: `Schema: {schema}`
      2. e.g. detailed `Please format your answer using the following schema: {schema}. Remember to validate all typing and formatting constraints. Do not format your answer in Markdown,`
3. Difficulty of Question
   1. The question exists only to serve as a proxy for eliciting a response worthy of output formatting. To focus the environment towards schema adherence, the question should be easy.
      1. e.g. simple: `Please provide a response based on the document and provided schema`.

For any parsed outputs, we use the `openapi-schema-validator` library for verification.

We currently have support for 5 different schema output types: JSON, YAML, XML, TOML, and CSV.

> [!IMPORTANT]
> Evaluation is only based on the **schema adherence** of the generated output.
> **The actual content of the generation is *not* verified**, thus it is advised that the task used for prompt creation is not too difficult for the model.


# Example usage

## Running servers
The following are example commands for running this resources server, along with the simple agent and an OpenAI model:
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
    +input_jsonl_fpath=resources_servers/structured_outputs/data/structured_outputs_260309_nano_v3_sdg_json_yaml_xml_val.jsonl \
    +output_jsonl_fpath=results/example_structured_outputs_json.jsonl \
    +resume_from_cache=True \
    +num_samples_in_parallel=256
```

You can see breakdown of results from the rollout file using the provided breakdown_metrics file.
```bash
python resources_servers/structured_outputs/misc/breakdown_rollouts_metrics.py \
   -f results/example_structured_outputs_json.jsonl
```

## Downloading Data
### Version 1 [251027] (JSON only)
You can prepare the data for training with:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_json.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/structured_outputs \
    +mode=train_preparation +should_download=true
```

### Version 2 [260310] (JSON, YAML, XML)
```bash
# prepare
export config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_json_yaml_xml_v1.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/structured_outputs/ \
    +mode=train_preparation \
    +should_download=true
```

### Version 3 [260331] (JSON, YAML, XML, TOML, CSV)
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_all_formats_v1.yaml"
ng_run "+config_paths=[${config_paths}]"
```

Collect rollouts:
```bash
ng_collect_rollouts \
    +agent_name=structured_outputs_all_formats_simple_agent \
    +input_jsonl_fpath=resources_servers/structured_outputs/data/ds1_schema_adherence_train.jsonl \
    +output_jsonl_fpath=results/structured_outputs_all_formats.jsonl \
    +resume_from_cache=True \
    +num_samples_in_parallel=256
```

# Testing
```
ng_test +entrypoint=resources_servers/structured_outputs
```

# Licensing information
Code: Apache 2.0

Data: CC BY 4.0

Dependencies
- nemo_gym: Apache 2.0
- openapi-schema-validator: [BSD-3-Clause license](https://github.com/python-openapi/openapi-schema-validator/blob/master/LICENSE)
- xmltodict: [MIT](https://github.com/martinblech/xmltodict/blob/master/LICENSE)
