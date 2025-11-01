# Description
This is a no-op resource server, which may be a helpful utility for collecting rollouts while deferring verification.

# Example usage

## Running servers
The following are example commands for running this resource server, along with the simple agent and a VLLM model:
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml, \
resources_servers/noop/configs/noop.yaml"
ng_run "+config_paths=[$config_paths]"
```

Then, rollouts can be collected using a command such as the following:
```bash
ng_collect_rollouts \
    +agent_name=noop_simple_agent \
    +input_jsonl_fpath=your_rollout_input_dataset.jsonl \
    +output_jsonl_fpath=your_rollout_output_responses.jsonl \
    +limit=10
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
