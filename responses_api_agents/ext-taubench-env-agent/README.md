# Description

## Prepare data
You must have the raw examples at `responses_api_agents/ext-taubench-env-agent/data/final_delivery_dual_control/*.json`
```bash
python responses_api_agents/ext-taubench-env-agent/dataset_preprocess.py
```

Upload datasets
```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=ext-taubench-env \
    +version=0.0.4 \
    +input_jsonl_fpath=responses_api_agents/ext-taubench-env-agent/data/train.jsonl
```

```bash
config_paths="responses_api_agents/ext-taubench-env-agent/configs/ext-taubench-env.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/ext-taubench-env \
    +mode=train_preparation \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.base_url=dummy' \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.api_key=dummy' \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.model=dummy'
```

## Run
Spin up server with OpenAIModel
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
responses_api_agents/ext-taubench-env-agent/configs/ext-taubench-env-openai-user.yaml"
ng_run "+config_paths=[$config_paths]"
```

Spin up server with VLLMModel
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
responses_api_agents/ext-taubench-env-agent/configs/ext-taubench-env.yaml"
ng_run "+config_paths=[$config_paths]" \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.base_url=<>' \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.api_key=<>' \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.model=<>'
```

Collect rollouts
```bash
ng_collect_rollouts \
    +agent_name=ext-taubench-env-agent \
    +input_jsonl_fpath=responses_api_agents/ext-taubench-env-agent/data/example.jsonl \
    +output_jsonl_fpath=results/ext-taubench-env-agent.jsonl \
    +num_samples_in_parallel=1 \
    +num_repeats=1 \
    +limit=1
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
