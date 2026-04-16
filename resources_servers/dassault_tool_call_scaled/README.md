# Description

Dassault scaled function calling benchmark. Measures function selection accuracy as the number of available functions grows from 20 to 100, with deliberate semantic overlap to stress-test model discrimination. 10 function clusters, 100 total functions, 250 test queries. All tool scales (20, 40, 60, 80, 100) are combined into a single dataset; each row's `verifier_metadata.function_count` identifies its scale for post-hoc aggregation.

Data links: ?

Dataset creation
```bash
# Preprocess dataset to Gym format (single combined JSONL with all tool scales)
python resources_servers/dassault_tool_call_scaled/preprocess_to_gym.py \
    --evals-dir /path/to/evals/scaled/test_data

# Upload to Gitlab
ng_upload_dataset_to_gitlab \
    +dataset_name=dassault_tool_call_scaled \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/dassault_tool_call_scaled/data/scaled.jsonl
```

Usage
```bash
# Put your policy information in env.yaml. Example for OpenAI GPT 4.1:
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: {your OpenAI API key}
policy_model_name: gpt-4.1-2025-04-14" > env.yaml

# Or for an NVIDIA API endpoint (uses vllm_model instead of openai_model):
# echo "policy_base_url: https://integrate.api.nvidia.com/v1
# policy_api_key: {your NVIDIA API key}
# policy_model_name: {model name}" > env.yaml

# Download data
config_paths="resources_servers/dassault_tool_call_scaled/configs/dassault_tool_call_scaled.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/dassault_tool_call_scaled \
    +mode=train_preparation \
    +should_download=true \
    +data_source=gitlab

# Spin up the servers in terminal 1
# Use openai_model for OpenAI endpoints, vllm_model for NVIDIA/vLLM endpoints
config_paths="resources_servers/dassault_tool_call_scaled/configs/dassault_tool_call_scaled.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# Collect rollouts in terminal 2
ng_collect_rollouts +agent_name=dassault_tool_call_scaled_simple_agent \
    +input_jsonl_fpath=resources_servers/dassault_tool_call_scaled/data/scaled.jsonl \
    +output_jsonl_fpath=resources_servers/dassault_tool_call_scaled/data/validation_rollouts.jsonl \
    +num_repeats=1 \
    +num_samples_in_parallel=4

# View rollouts
ng_viewer +jsonl_fpath=resources_servers/dassault_tool_call_scaled/data/validation_rollouts.jsonl
```

Scores
TBD


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
