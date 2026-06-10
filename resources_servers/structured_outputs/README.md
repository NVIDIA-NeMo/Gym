# Description
> Keywords: Instruction Following, Structured Outputs, Schema Adherence

This resources server verifies whether a model can follow structured-output
schema constraints. It supports two response surfaces:

- Text output, v1-v3: the schema is shown in the prompt and the model emits
  JSON, YAML, XML, TOML, or CSV text.
- Tool-call output, v4: the schema is supplied as an OpenAI Responses API
  function tool in `responses_create_params.tools`, and the model's final
  answer is a function call.

Text-output problems consist of three components:
1. Document
2. Output Formatting Instruction (Schema)
3. Question

Tool-call rows keep the prompt schema-free. The model sees a document, a short
instruction, and one or more tool schemas.

The original text-output dataset can be found at
https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs.


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

We currently support text outputs in JSON, YAML, XML, TOML, and CSV, plus
tool-call structured outputs in v4.

> [!IMPORTANT]
> Evaluation is only based on the **schema adherence** of the generated output.
> **The actual content of the generation is *not* verified**, thus it is advised that the task used for prompt creation is not too difficult for the model.

## Design Notes

Reusable guidance for designing structured-output data and verifiers across
text-output and tool-call variants is in
[misc/structured-outputs-design.md](misc/structured-outputs-design.md).


# Example usage

## Running servers
The following command runs the text-output JSON config with the simple agent and
an OpenAI model:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/structured_outputs/configs/structured_outputs_json.yaml"
ng_run "+config_paths=[$config_paths]"
```

Then, text-output rollouts can be collected using a command such as:
```bash
ng_collect_rollouts \
    +agent_name=structured_outputs_simple_agent \
    +input_jsonl_fpath=resources_servers/structured_outputs/data/structured_outputs_260309_nano_v3_sdg_json_yaml_xml_val.jsonl \
    +output_jsonl_fpath=results/example_structured_outputs_json.jsonl \
    +resume_from_cache=True \
    +num_samples_in_parallel=256
```

For v4 tool-call structured outputs, use `structured_outputs_v4.yaml` and the
v4 simple agent. The config routes through a non-executing agent because the
emitted function call is the final answer, not an action to execute:
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_v4.yaml"
ng_run "+config_paths=[${config_paths}]"
```

```bash
ng_collect_rollouts \
    +agent_name=structured_outputs_v4_simple_agent \
    +input_jsonl_fpath=resources_servers/structured_outputs/data/structured_outputs_v4_tool_call.jsonl \
    +output_jsonl_fpath=results/example_structured_outputs_v4_tool_call.jsonl \
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
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_json_yaml_xml_v1.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/structured_outputs/ \
    +mode=train_preparation \
    +should_download=true
```

### Version 3 [260409] (JSON, YAML, XML, TOML, CSV)
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_v3.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/structured_outputs_v3/ \
    +mode=train_preparation \
    +should_download=true
```

### Version 4 [260424] (Tool-call structured outputs)
Version 4 generates prompts where the schema is provided as an OpenAI Responses
API function tool instead of prompt text. The prompt contains only the document
and a short extraction instruction.

The uploaded GitLab dataset is:

- `dataset_name`: `structured_outputs_v4_tool_call`
- `version`: `0.0.1`
- `artifact_fpath`: `structured_outputs_v4_tool_call.jsonl`

Prepare the v4 training data with:
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/structured_outputs/configs/structured_outputs_v4.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/structured_outputs_v4_tool_call/ \
    +mode=train_preparation \
    +should_download=true
```

The v4 config wires the train dataset to
`resources_servers/structured_outputs/data/structured_outputs_v4_tool_call.jsonl`
and the example dataset to
`resources_servers/structured_outputs/data/structured_outputs_v4_example.jsonl`.

For v4, the config uses the non-executing simple agent because the tool call is
the final answer being verified, not an action that should be executed by the
agent. Rows use `tool_choice: auto`, and some rows allow `parallel_tool_calls`
for coverage, but the reward contract still requires exactly one final function
call. Missing, multiple, wrong, or malformed tool calls receive zero reward. The
verifier finds the matching function call, JSON-decodes its arguments, unwraps
`tool_payload_key` when the row uses a wrapper mode, and validates the payload
against `schema_str`.

Generation details and CLI examples are in
`resources_servers/structured_outputs/misc/data_generation/structured_outputs_v4/README.md`.

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
- tomli-w: [MIT](https://github.com/hukkin/tomli-w/blob/master/LICENSE)
- xmltodict: [MIT](https://github.com/martinblech/xmltodict/blob/master/LICENSE)
