# Reasoning Gym Resources Server

Integration of reasoning gym: https://github.com/open-thought/reasoning-gym

From reasoning gym's readme, "It currently provides more than 100 tasks over many domains, including but not limited to algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and many common games."

# Dataset prep

**Single task:**
```bash
python scripts/create_dataset.py \
    --task knights_knaves \
    --size 500 \
    --seed 42 \
    --output data/train_knights_knaves.jsonl
```

**Multiple tasks (composite):**
```bash
python scripts/create_dataset.py \
    --tasks knights_knaves,syllogisms,leg_counting \
    --size 1000 \
    --output data/train_composite.jsonl
```

**All tasks in a category:**
```bash
python scripts/create_dataset.py \
    --category logic \
    --size 1000 \
    --output data/train_logic.jsonl
```


**All tasks in all categories:**
```bash
python scripts/create_dataset.py \
    --all-tasks \
    --size 1000 \
    --output data/train_all.jsonl
```


**With custom config:**
```bash
python scripts/create_dataset.py \
    --task knights_knaves \
    --size 500 \
    --config '{"n_people": 3, "depth_constraint": 3}' \
    --output data/train_hard.jsonl
```

# Rollout collection

## Start a vllm server

```bash
pip install -U "vllm>=0.12.0"
 
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
 
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
 --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 10240 \
  --trust-remote-code \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

## Create env.yaml:


```
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## Launch nemo gym servers

```bash
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_simple_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=results/reasoning_gym_rollouts.jsonl \
    +limit=5
```