(multi-verifier-rollouts)=

# Multi-verifier rollouts
Gym is explicitly designed to support multi-verifier rollouts and training.

Let's say you want to use both the example_single_tool_call and example_multi_step training environments. Normally how you spin up the servers individually is:

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

If you want to use them both you would just add the yamls together like:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_run "+config_paths=[$config_paths]"
```

Build a dataset that contains data that hits both servers. Here, we add the agent ref used to route requests to the right agent server to the data.
```bash
jq -c '. + {"agent_ref": {"name": "example_single_tool_call_simple_agent"}}' resources_servers/example_single_tool_call/data/example.jsonl >> results/test_multiverifier_input.jsonl
jq -c '. + {"agent_ref": {"name": "example_multi_step_simple_agent"}}' resources_servers/example_multi_step/data/example.jsonl >> results/test_multiverifier_input.jsonl
```

And then run rollout collection like normal!
```bash
ng_collect_rollouts \
    +input_jsonl_fpath=results/test_multiverifier_input.jsonl \
    +output_jsonl_fpath=results/test_multiverifier_outputs.jsonl
```

Inside `results/test_multiverifier_outputs.jsonl`, you should see 10 rows with appropriate responses for each row.

The same process goes for data preparation and downstream training framework Gym configuration, you would just add additional server configs.
