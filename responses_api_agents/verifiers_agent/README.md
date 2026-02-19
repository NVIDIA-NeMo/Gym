# Description

This agent enables running Prime Intellect [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments, including many in Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections) in NeMo Gym. 

## Install Gym

```
git clone https://github.com/NVIDIA-NeMo/Gym
cd Gym
uv venv
source .venv/bin/activate
uv sync
```

## Test acereason-math example 

First set `env.yaml` for a local model:
```
policy_base_url: "http://localhost:8000/v1"
policy_api_key: <>
policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
```

Next, serve the model. 

Make sure to serve the model with longer context length than the generation length in your agent config (e.g. acereason-math.yaml)

```
uv pip install vllm 
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
```


Now launch NeMo Gym servers:
```
uv sync
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/verifiers_acereason-math.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Collect rollouts
```
ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl \
    +limit=5
```

View a rollout in the terminal
```
tail -n 1 responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl | jq | less
```


## Testing new prime environments from environments hub

Some examples: `primeintellect/acereason-math`, `primeintellect/ascii-tree` and `primeintellect/alphabet-sort`.

### Install an environment

```
uv add verifiers
uv add tool prime 
prime env install primeintellect/ascii-tree
```

### Creating a dataset

A helper script to make a dataset is in `scripts/create_datset.py`.

```
python3 scripts/create_dataset.py --env-id primeintellect/ascii-tree --size 5 --output data/ascii-tree-example.jsonl
```

### Update agent server requirements

Update `requirements.txt` to: 
```
-e nemo-gym[dev] @ ../../
verifiers==0.1.9.post3
--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
ascii-tree
```
### Update agent config
Create `configs/ascii-tree.yaml`, primarily updating env id, and any other env specific args: 
```
verifiers_agent:
  responses_api_agents:
    verifiers_agent:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      model_name: ""
      vf_env_id: ascii-tree
      vf_env_args: {}
      group_size: 1
      max_concurrent_generation: -1
      max_concurrent_scoring: -1
      max_tokens: 8192
      temperature: 1.0
      top_p: 1.0

```

Now launch NeMo Gym servers:
```
uv sync
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/ascii-tree.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Collect rollouts
```
ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/ascii-tree-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/ascii-tree-example-rollouts.jsonl \
    +limit=5
```


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- verifiers: Apache 2.0
