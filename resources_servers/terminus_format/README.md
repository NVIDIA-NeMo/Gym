# Description

This is a simple json format verifier environment intended to teach the model how to follow output formatting instructions.

# Data

The dataset was created by taking terminal-bench sft trajectories and slicing them into incremental step-by-step prefixes, each turned into a prompt asking the model for the next action, with outputs validated only for formatting correctness.

Data: [Gitlab model registry link](https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/148/versions/170#/)

Download this artifact:
```bash
ng_download_dataset_from_gitlab \
    +dataset_name=terminus_format_dataset \
    +version=0.0.1 \
    +artifact_fpath=example.jsonl \
    +output_fpath=resources_servers/terminus_format/data/example.jsonl
```

Example data:
`resources_servers/terminus_format/data/example.jsonl`

# Licensing information
?

# Example usage

The following are example commands for running this resource server, along with the simple agent and an OpenAI model:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/terminus_format/configs/terminus_format.yaml"
ng_run "+config_paths=[$config_paths]"
```

Then, rollouts can be collected using a command such as the following:
```bash
ng_collect_rollouts \
    +agent_name=terminus_format_simple_agent \
    +input_jsonl_fpath=resources_servers/terminus_format/data/example.jsonl \
    +output_jsonl_fpath=results/example_terminus_format_json.jsonl \
    +limit=1
```

Dependencies
- nemo_gym: Apache 2.0
- openapi-schema-validator: [BSD-3-Clause license](https://github.com/python-openapi/openapi-schema-validator/blob/master/LICENSE)


# Next Steps

- Add more template schemas
- Add more generalisable/diverse data
- Add stricter format validation during verification
