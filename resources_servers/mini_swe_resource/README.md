# Description

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/mini_swe_resource/configs/mini_swe_resource.yaml"
ng_run "+config_paths=[$config_paths]" \
    ++mini_swe_main_agent_train.responses_api_agents.mini_swe_agent.cache_dir_template={your cache dir} \
    ++mini_swe_main_agent_validation.responses_api_agents.mini_swe_agent.cache_dir_template={your cache dir}
```

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
