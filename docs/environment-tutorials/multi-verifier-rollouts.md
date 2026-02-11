(multi-verifier-rollouts)=

# Multi-verifier rollouts
Gym is explicitly designed to support multi-verifier rollouts and training.

Let's say you want to use both math and search verifiers. Normally how you spin up the servers individually is:
For math:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[${config_paths}]"
```
For search:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

If you want to use them both you would just add the yamls together like:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

The same process goes for data preparation and downstream training framework Gym configuration, you would just add additional server configs.
