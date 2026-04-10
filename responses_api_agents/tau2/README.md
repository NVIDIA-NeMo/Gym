# Description
```bash
config_paths="benchmarks/tau2/config.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++nemo_gym_log_dir=results/tau2 \
    '++gpt-5_2-2025-12-11.responses_api_models.openai_model.openai_api_key=${openai_api_key}' \
    '++gpt-5_2-2025-12-11.responses_api_models.openai_model.extra_body._delete_key=max_output_tokens'
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
