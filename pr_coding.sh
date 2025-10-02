
# Download train data
ng_download_dataset_from_gitlab \
    +dataset_name=opencodereasoning_filtered \
    +version=0.0.1 \
    +artifact_fpath=opencodereasoning_filtered_25k_train.jsonl \
    +output_fpath=resources_servers/comp_coding/data/train.jsonl

ng_download_dataset_from_gitlab \
    +dataset_name=livecodebench \
    +version=0.0.1 \
    +artifact_fpath=livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl \
    +output_fpath=resources_servers/comp_coding/data/validation.jsonl

# Parallel reasoning
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/comp_coding/configs/parallel_reasoning.yaml"
ng_run "+config_paths=[$config_paths]" \
    +comp_coding.resources_servers.comp_coding.judge_model_server.name=policy_model &


ng_collect_rollouts \
    +agent_name=parallel_reasoning_simple_agent \
    +input_jsonl_fpath=resources_servers/comp_coding/data/validation.jsonl \
    +output_jsonl_fpath=results/parallel_reasoning_trajectory_collection_limit_debug_1002.jsonl \
    +limit=5


# Serial Competitive Coding
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/comp_coding/configs/comp_coding.yaml"
ng_run "+config_paths=[$config_paths]" \
    +comp_coding.resources_servers.comp_coding.judge_model_server.name=policy_model &


ng_collect_rollouts \
    +agent_name=comp_coding_simple_agent \
    +input_jsonl_fpath=resources_servers/comp_coding/data/train.jsonl \
    +output_jsonl_fpath=results/comp_coding_trajectory_collection_limit_5.jsonl \
    +limit=5