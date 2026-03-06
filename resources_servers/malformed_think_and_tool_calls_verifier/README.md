# Description
This is a resource server that is a proxy for checking if the model is able to think properly and format tool calls properly.

There are two error profiles that it looks for in the data:
Rules:
1) malformed_thinking:
    - message["type"] == "reasoning"
    - "<tool_call>" or "</tool_call>" appears in any summary[i]["text"]
2) malformed_tool_call:
    - message["type"] == "message"
    - message["role"] == "assistant"
    - "<tool_call>" or "</tool_call>" appears in any content[i]["text"]

The validation set contains 500 samples that may lead to malformed_thinking, and 98 samples that may lead to malformed_tool_call. The goal of the model is to not generate an erroneous completion.

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0


# Dataset Commands
```
# upload
ng_upload_dataset_to_gitlab \
    +dataset_name=malformed_think_and_tool_calls_verifier_super \
    +version=0.0.1 \
    +input_jsonl_fpath=data/malformed_think_and_tool_calls_verifier/260303_full_outputs_max_subset.jsonl

# download
ng_download_dataset_from_gitlab \
    +dataset_name=malformed_think_and_tool_calls_verifier_super \
    +version=0.0.1 \
    +artifact_fpath=260303_full_outputs_max_subset.jsonl \
    +output_fpath=data/malformed_think_and_tool_calls_verifier/260303_full_outputs_max_subset.jsonl


# example preparation
ng_prepare_data "+config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/malformed_think_and_tool_calls_verifier/configs/malformed_think_and_tool_calls_verifier.yaml]" \
    +output_dirpath=data/malformed_think_and_tool_calls_verifier \
    +mode=example_validation


# validation data preparation
ng_prepare_data "+config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/malformed_think_and_tool_calls_verifier/configs/malformed_think_and_tool_calls_verifier.yaml]" \
    +output_dirpath=data/malformed_think_and_tool_calls_verifier \
    +mode=train_preparation \
    +should_download=true
```

# Validation Commands
```
# Spin up server
echo "Spin up server"
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml, \
resources_servers/malformed_think_and_tool_calls_verifier/configs/malformed_think_and_tool_calls_verifier.yaml"
ng_run "+config_paths=[$config_paths]" \
    +policy_model_name=$SERVED_MODEL_NAME \
    +policy_base_url=$MODEL_BASE_URL \
    +policy_api_key=$MODEL_API_KEY &

# Rollout
mkdir -p "rollouts/malformed_think_and_tool_calls_verifier"
export OUTPUT_FPATH="rollouts/malformed_think_and_tool_calls_verifier/${SERVED_MODEL_NAME##*/}.jsonl"
ng_collect_rollouts \
    +agent_name=malformed_think_and_tool_calls_verifier_agent \
    +input_jsonl_fpath=resources_servers/malformed_think_and_tool_calls_verifier/data/validation_prepare.jsonl \
    +output_jsonl_fpath=$OUTPUT_FPATH \
    +resume_from_cache=True \
    +num_samples_in_parallel=64

# Run validation metrics breakdown script
python resources_servers/malformed_think_and_tool_calls_verifier/scripts/breakdown_metrics.py \
    -f rollouts/malformed_think_and_tool_calls_verifier/{SERVED_MODEL_NAME##*/}.jsonl
```

# Testing
```
ng_test +entrypoint=resources_servers/malformed_think_and_tool_calls_verifier
```