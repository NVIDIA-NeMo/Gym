# Description

Dataset creation
```bash
# Preprocess dataset to Gym format
python resources_servers/servicenow_document_reasoning/preprocess_to_gym.py

# Upload to Gitlab
ng_upload_dataset_to_gitlab \
    +dataset_name=servicenow_document_reasoning \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/servicenow_document_reasoning/data/validation.jsonl

# Upload rollout datasets to Gitlab
ng_upload_dataset_to_gitlab \
    +dataset_name=servicenow_document_reasoning \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/servicenow_document_reasoning/data/gpt4p1_validation_rollouts.jsonl
```

Usage
```bash
# Put your policy information in env.yaml. Example for OpenAI GPT 4.1:
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: {your OpenAI API key}
policy_model_name: gpt-4.1-2025-04-14" > env.yaml

# Put your OpenAI API key in env.yaml
echo "servicenow_document_reasoning_judge_model:
  responses_api_models:
    openai_model:
      openai_api_key: {your OpenAI API key}" >> env.yaml

# Download data
config_paths="resources_servers/servicenow_document_reasoning/configs/servicenow_document_reasoning.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/servicenow_document_reasoning \
    +mode=train_preparation \
    +should_download=true

# Spin up the servers in terminal 1
config_paths="resources_servers/servicenow_document_reasoning/configs/servicenow_document_reasoning.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# Collect rollouts in terminal 2
ng_collect_rollouts +agent_name=servicenow_document_reasoning_simple_agent \
    +input_jsonl_fpath=data/servicenow_document_reasoning/validation.jsonl \
    +output_jsonl_fpath=resources_servers/servicenow_document_reasoning/data/validation_rollouts.jsonl

# View rollouts
ng_viewer +jsonl_fpath=resources_servers/servicenow_document_reasoning/data/validation_rollouts.jsonl
```

Scores (using GPT 4.1 as the judge). We calculate mean@8 score since this is a knowledge task. See https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/91/versions/100#/ for the rollouts themselves.
GPT 4.1
```json
{
    "reward": 0.7625,
    "unanswerable": 0.0,
    "matches_reference": 0.7625
}
```
GPT 5
```json
{
    "reward": 0.8777777777777778,
    "unanswerable": 0.0,
    "matches_reference": 0.8777777777777778
}
```

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
