# Description

Dataset creation
```bash
# Preprocess dataset to Gym format
python resources_servers/cohesity_netbackup_rag/preprocess_to_gym.py

# Upload to Gitlab
ng_upload_dataset_to_gitlab \
    +dataset_name=cohesity_netbackup_rag \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/cohesity_netbackup_rag/data/validation.jsonl

# Upload rollout datasets to Gitlab
ng_upload_dataset_to_gitlab \
    +dataset_name=cohesity_netbackup_rag \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/cohesity_netbackup_rag/data/gpt4p1_validation_rollouts.jsonl
```

Usage
```bash
# Put your policy information in env.yaml. Example for OpenAI GPT 4.1:
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: {your OpenAI API key}
policy_model_name: gpt-4.1-2025-04-14" > env.yaml

# Put your OpenAI API key in env.yaml
echo "cohesity_netbackup_rag_judge_model:
  responses_api_models:
    openai_model:
      openai_api_key: {your OpenAI API key}" >> env.yaml

# Download data
config_paths="resources_servers/cohesity_netbackup_rag/configs/cohesity_netbackup_rag.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/cohesity_netbackup_rag \
    +mode=train_preparation \
    +should_download=true

# Spin up the servers in terminal 1
config_paths="resources_servers/cohesity_netbackup_rag/configs/cohesity_netbackup_rag.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# Collect rollouts in terminal 2
ng_collect_rollouts +agent_name=cohesity_netbackup_rag_simple_agent \
    +input_jsonl_fpath=data/cohesity_netbackup_rag/validation.jsonl \
    +output_jsonl_fpath=resources_servers/cohesity_netbackup_rag/data/validation_rollouts.jsonl

# View rollouts
ng_viewer +jsonl_fpath=resources_servers/cohesity_netbackup_rag/data/validation_rollouts.jsonl
```

Scores (using GPT 4.1 as the judge). We calculate mean@8 score since this is a knowledge task. See https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/91/versions/100#/ for the rollouts themselves.
GPT 4.1
```json
{
    "reward": 0.7037037037037037,
    "matches_reference_content": 0.9814814814814815,
    "contains_additional_content_not_in_context": 0.2777777777777778
}
```
GPT 5
```json
{
    "reward": 0.8148148148148148,
    "matches_reference_content": 1.0,
    "contains_additional_content_not_in_context": 0.18518518518518517
}
```


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
