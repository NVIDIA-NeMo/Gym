# Description

Data links: ?

Dataset creation
```bash
# Preprocess dataset to Gym format
python resources_servers/crowdstrike_logscale_syntax/preprocess_to_gym.py

# Upload to Gitlab
ng_upload_dataset_to_gitlab \
    +dataset_name=crowdstrike_logscale_syntax \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/crowdstrike_logscale_syntax/data/validation.jsonl

# Upload rollout datasets to Gitlab
ng_upload_dataset_to_gitlab \
    +dataset_name=crowdstrike_logscale_syntax \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/crowdstrike_logscale_syntax/data/gpt4p1_validation_rollouts.jsonl
```

Usage
```bash
# Put your policy information in env.yaml. Example for OpenAI GPT 4.1:
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: {your OpenAI API key}
policy_model_name: gpt-4.1-2025-04-14" > env.yaml

# Put your OpenAI API key in env.yaml
echo "crowdstrike_logscale_syntax_judge_model:
  responses_api_models:
    openai_model:
      openai_api_key: {your OpenAI API key}" >> env.yaml

# Download data
config_paths="resources_servers/crowdstrike_logscale_syntax/configs/crowdstrike_logscale_syntax.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/crowdstrike_logscale_syntax \
    +mode=train_preparation \
    +should_download=true

# Spin up the servers in terminal 1
config_paths="resources_servers/crowdstrike_logscale_syntax/configs/crowdstrike_logscale_syntax.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# Collect rollouts in terminal 2
ng_collect_rollouts +agent_name=crowdstrike_logscale_syntax_simple_agent \
    +input_jsonl_fpath=data/crowdstrike_logscale_syntax/validation.jsonl \
    +output_jsonl_fpath=resources_servers/crowdstrike_logscale_syntax/data/validation_rollouts.jsonl

# View rollouts
ng_viewer +jsonl_fpath=resources_servers/crowdstrike_logscale_syntax/data/validation_rollouts.jsonl
```

Scores (using GPT 4.1 as the judge). We calculate mean@8 score since this is a knowledge task. See https://gitlab-master.nvidia.com/bxyu/nemo-gym/-/ml/models/91/versions/100#/ for the rollouts themselves.
GPT 4.1
```json
{
    "reward": 0.3638211382113821,
    "correct_syntax": 0.40447154471544716,
    "used_keyword_arguments": 0.6585365853658537
}
```
GPT 5
```json
{
    "reward": 0.3861788617886179,
    "correct_syntax": 0.40752032520325204,
    "used_keyword_arguments": 0.7276422764227642
}
```


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
