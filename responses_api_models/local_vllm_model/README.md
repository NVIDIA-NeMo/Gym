# E2E sanity testing
```bash
config_paths="responses_api_models/local_vllm_model/configs/qwen3_235b_a22b_instruct_2507.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++qwen3_235b_a22b_instruct_2507_model_server.responses_api_models.local_vllm_model.model=trl-internal-testing/tiny-Qwen3ForCausalLM
```
