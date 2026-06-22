# Description


# End-to-end test with GPT OSS 20B reasoning high
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml,\
responses_api_models/local_vllm_model_proxy/configs/local_vllm_model_proxy.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++policy_model_proxy.responses_api_models.local_vllm_model_proxy.model_server.name=gpt-oss-20b-reasoning-high \
    ++policy_model_proxy.responses_api_models.local_vllm_model_proxy.extra_body.max_tokens=10
```


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
