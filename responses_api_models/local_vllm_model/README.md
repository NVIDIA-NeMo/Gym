- [Example run config](#example-run-config)
- [E2E sanity testing](#e2e-sanity-testing)
  - [1 node](#1-node)
    - [1 instance of 1x8](#1-instance-of-1x8)
    - [1 instance of 2x4](#1-instance-of-2x4)
  - [2 nodes](#2-nodes)
    - [2 instances of 1x8](#2-instances-of-1x8)
    - [2 instances of 2x4](#2-instances-of-2x4)
    - [1 instance of 1x16](#1-instance-of-1x16)
  - [4 nodes](#4-nodes)
    - [1 instance of 2x16](#1-instance-of-2x16)
    - [2 instances 1x16](#2-instances-1x16)
  - [8 nodes](#8-nodes)
    - [2 instances 2x16](#2-instances-2x16)

# Example run config
Run this on a single GPU node! Set tensor_parallel_size * data_parallel_size to the number of GPUs on your node. For this single node config, data_parallel_size_local is equal to data_parallel_size

```bash
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/local_vllm_model/configs/nano_v3_single_node.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4 \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size_local=2 &> temp.log &
```

View the logs
```bash
tail -f temp.log
```

Call the server. If you see a model response here, then everything is working as intended!
```bash
python responses_api_agents/simple_agent/client.py
```


# E2E sanity testing

## 1 node
### 1 instance of 1x8
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### 1 instance of 2x4
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4
```

## 2 nodes
### 2 instances of 1x8
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml,\
responses_api_models/local_vllm_model/configs/openai/gpt-oss-120b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### 2 instances of 2x4
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml,\
responses_api_models/local_vllm_model/configs/openai/gpt-oss-120b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4 \
    ++gpt-oss-120b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++gpt-oss-120b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4
```

### 1 instance of 1x16
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=16
```

## 4 nodes
### 1 instance of 2x16
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=16
```

### 2 instances 1x16
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml,\
responses_api_models/local_vllm_model/configs/openai/gpt-oss-120b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=16 \
    ++gpt-oss-120b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=16
```

## 8 nodes
### 2 instances 2x16
```bash
config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml,\
responses_api_models/local_vllm_model/configs/openai/gpt-oss-120b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=16 \
    ++gpt-oss-120b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=2 \
    ++gpt-oss-120b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=16
```
