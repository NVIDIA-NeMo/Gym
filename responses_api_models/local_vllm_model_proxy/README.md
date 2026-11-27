# Description


# End-to-end test with GPT OSS 20B reasoning high
```bash
gym env start \
    --model-type local_vllm_model/openai/gpt-oss-20b-reasoning-high \
    --model-type local_vllm_model_proxy \
    ++policy_model_proxy.responses_api_models.local_vllm_model_proxy.model_server.name=gpt-oss-20b-reasoning-high \
    ++policy_model_proxy.responses_api_models.local_vllm_model_proxy.extra_body.max_tokens=10
```


# Bounding startup
By default the proxy waits without limit for the referenced `local_vllm_model` server to come up. Set
`model_server_wait_timeout_seconds` to make startup fail with a `TimeoutError` after that long. The same value is the
time limit for the proxy's config request to that server.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
