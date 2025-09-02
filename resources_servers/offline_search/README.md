Setup Server:
```
config_paths_str="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/offline_search/configs/offline_search.yaml"

ng_run "+config_paths=[$config_paths_str]" "+simple_agent.responses_api_agents.simple_agent.resources_server.name=offline_search_resources_server"

```
