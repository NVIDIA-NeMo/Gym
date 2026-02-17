(multi-environment)=

# Multi-Environment Training
NeMo Gym supports training on multiple environments simultaneously. Multi-verifier training is another term for this concept.

## Why Train on Multiple Environments?
This technique often results in more stable gains across multiple benchmarks. Single-environment training may cause unrecoverable degradation of other benchmarks.

## How to Configure
Suppose you want to use both the example_single_tool_call and example_multi_step training environments. To start each server individually:

For example_single_tool_call:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml"
ng_run "+config_paths=[${config_paths}]"
```

For example_multi_step:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_run "+config_paths=[$config_paths]"
```

To use both environments, add the YAML configs together as follows:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Dataset Preparation

Build a dataset that contains data for both servers. Add the agent ref used to route requests to the correct agent server to each record.
```bash
jq -c '. + {"agent_ref": {"name": "example_single_tool_call_simple_agent"}}' resources_servers/example_single_tool_call/data/example.jsonl >> results/test_multiverifier_input.jsonl
jq -c '. + {"agent_ref": {"name": "example_multi_step_simple_agent"}}' resources_servers/example_multi_step/data/example.jsonl >> results/test_multiverifier_input.jsonl
```

## Rollout Collection

Run rollout collection as usual.
```bash
ng_collect_rollouts \
    +input_jsonl_fpath=results/test_multiverifier_input.jsonl \
    +output_jsonl_fpath=results/test_multiverifier_outputs.jsonl
```

Inside `results/test_multiverifier_outputs.jsonl`, you should see 10 rows with appropriate responses for each row.

Apply the same process for data preparation and downstream training. Add additional server configs as needed.
