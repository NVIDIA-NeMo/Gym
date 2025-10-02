# Reasoning Gym Resources Server

Integration of reasoning gym: https://github.com/open-thought/reasoning-gym

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

**With custom config:**
```bash
python scripts/create_dataset.py \
    --task knights_knaves \
    --size 500 \
    --config '{"n_people": 3, "depth_constraint": 3}' \
    --output data/train_hard.jsonl
```

```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```

```bash
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_simple_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=results/reasoning_gym_rollouts.jsonl \
    +limit=5
```