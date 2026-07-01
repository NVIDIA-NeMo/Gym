# Reasoning Gym

Integration of reasoning gym: https://github.com/open-thought/reasoning-gym

From reasoning gym's readme, "It currently provides more than 100 tasks over many domains, including but not limited to algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and many common games."

# Dataset prep

Download the pre-built training dataset from HuggingFace:

```bash
python environments/reasoning_gym/prepare.py
```

Or generate a custom dataset using the create_dataset script:

**Single task:**
```bash
python environments/reasoning_gym/scripts/create_dataset.py \
    --task knights_knaves \
    --size 500 \
    --seed 42 \
    --output environments/reasoning_gym/data/train_knights_knaves.jsonl
```

**Multiple tasks (composite):**
```bash
python environments/reasoning_gym/scripts/create_dataset.py \
    --tasks knights_knaves,syllogisms,leg_counting \
    --size 1000 \
    --output environments/reasoning_gym/data/train_composite.jsonl
```

**All tasks in a category:**
```bash
python environments/reasoning_gym/scripts/create_dataset.py \
    --category logic \
    --size 1000 \
    --output environments/reasoning_gym/data/train_logic.jsonl
```

**All tasks in all categories:**
```bash
python environments/reasoning_gym/scripts/create_dataset.py \
    --all-tasks \
    --size 1000 \
    --output environments/reasoning_gym/data/train_all.jsonl
```

**With custom config:**
```bash
python environments/reasoning_gym/scripts/create_dataset.py \
    --task knights_knaves \
    --size 500 \
    --config '{"n_people": 3, "depth_constraint": 3}' \
    --output environments/reasoning_gym/data/train_hard.jsonl
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

## Create env.yaml

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## Launch nemo gym servers

```bash
gym env start --environment reasoning_gym --model-type vllm_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent reasoning_gym_simple_agent \
    --input environments/reasoning_gym/data/example.jsonl \
    --output results/reasoning_gym_rollouts.jsonl \
    --limit 5
```
