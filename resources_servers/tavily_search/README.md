# Description
RL enviroment which allows access to web search (Search Provider: Tavily)

## Key config parameters
- You will need to create a Tavily key. Please reach out to @rgala to get access.
- Legal requires us to exclude some domains which we are not allowed to search. Please pass the file below (on DFW, LAX) before using the environment
- This environment uses LLM-as-judge to gauge the correctness of answers. Recommended judge model is Qwen3-235B-A22B-Instruct-2507.


Required to add env.yaml / your config
```

search_judge_model_base_url: <YOUR_JUDGE_MODEL_URL>
search_judge_model_api_key: ""
search_judge_model_name: Qwen/Qwen3-235B-A22B-Instruct-2507

tavily_search_resources_server:
  resources_servers:
    tavily_search:
      tavily_api_key: <YOUR_KEY>
      exclude_domains_file_path: /lustre/fsw/portfolios/llmservice/users/rgala/frozen/2025_12_15_nv_tdm_opt_out_registry.json
```





Commands to Run
```
ng_download_dataset_from_gitlab \
    +dataset_name=tavily_search \
    +version=0.0.1 \
    +artifact_fpath=sft_samples_train.jsonl \
    +output_fpath=resources_servers/tavily_search/data/sft_samples/sft_samples_train.jsonl

ng_download_dataset_from_gitlab \
    +dataset_name=tavily_search \
    +version=0.0.1 \
    +artifact_fpath=sft_samples_validation.jsonl \
    +output_fpath=resources_servers/tavily_search/data/sft_samples/sft_samples_validation.jsonl

config_paths="resources_servers/tavily_search/configs/tavily_search_judge_vllm_model.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[${config_paths}]"
```


### Performance Metrics
100*16 samples:
- Acc: 0.3212
- Time in ng_collect: 44 mins


# Licensing information
Code: ?
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0

