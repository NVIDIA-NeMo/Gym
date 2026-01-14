# Example run config
Run this on a single GPU node! Set tensor_parallel_size * data_parallel_size to the number of GPUs on your node. For this single node config, data_parallel_size_local is equal to data_parallel_size

```bash
config_paths="resources_servers/math_with_judge/configs/math_with_judge.yaml,\
responses_api_models/local_vllm_model/configs/nano_v3_single_node.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4 \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size_local=2
```

# E2E sanity testing
```bash
config_paths="responses_api_models/local_vllm_model/configs/qwen3_235b_a22b_instruct_2507.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++qwen3_235b_a22b_instruct_2507_model_server.responses_api_models.local_vllm_model.model=trl-internal-testing/tiny-Qwen3ForCausalLM \
    ++qwen3_235b_a22b_instruct_2507_model_server.responses_api_models.local_vllm_model.vllm_serve_kwargs.max_model_len=1024 \
    ++qwen3_235b_a22b_instruct_2507_model_server.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=1
```
