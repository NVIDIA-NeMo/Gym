# TextWorld Resources Server

Integrates: https://github.com/microsoft/TextWorld

Native multi-turn text adventure environments for RL training.

## Quick Start

### 1. Generate Dataset

```bash
python3 resources_servers/textworld/scripts/generate_games.py --workers 32
# Output: resources_servers/textworld/games/ with train/val/test splits
```

### 2. Create Training Examples

```bash
python3 resources_servers/textworld/scripts/create_examples.py \
  --all --split train \
  --output resources_servers/textworld/data/train.jsonl

python3 resources_servers/textworld/scripts/create_examples.py \
  --all --split val \
  --output resources_servers/textworld/data/val.jsonl

python3 resources_servers/textworld/scripts/create_examples.py \
  --all --split test \
  --output resources_servers/textworld/data/test.jsonl
```

### 3. Start Servers

```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240

ng_run "+config_paths=[resources_servers/textworld/configs/textworld.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

### 4. Collect Rollouts

```bash
ng_collect_rollouts +agent_name=textworld_simple_agent \
  +input_jsonl_fpath=resources_servers/textworld/data/train.jsonl \
  +output_jsonl_fpath=resources_servers/textworld/data/rollouts_train.jsonl \
  +limit=5
```

### 5. View Results

```bash
ng_viewer +jsonl_fpath=resources_servers/textworld/data/rollouts_example.jsonl
```

## Testing

```bash
ng_test +entrypoint=resources_servers/textworld
```

## Validation

```bash
python3 resources_servers/textworld/scripts/validate_dataset.py resources_servers/textworld/games
```

