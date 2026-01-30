# ARC-AGI resources server

Launch local vllm server
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

Set env.yaml in `Gym/`: 
```
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Create datasets: 
```
cd Gym/

# ARC-AGI-1
git clone https://github.com/fchollet/ARC-AGI
cd resources_servers/arc_agi
python3 create_dataset.py

# ARC-AGI-2
git clone https://github.com/arcprize/ARC-AGI-2
cd resources_servers/arc_agi
python3 create_dataset.py --version 2
```

Install Gym:
```
cd Gym/ 
uv venv 
source .venv/bin/activate
uv sync 
```

Start ARC-AGI environment (we can reuse the same one for ARC-AGI-1 and 2):
```bash
ng_run "+config_paths=[resources_servers/arc_agi/configs/arc_agi.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```


Collect rollouts:

ARC-AGI-1 example rollouts:
```bash
ng_collect_rollouts +agent_name=arc_agi_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/example_1.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/example_1_rollouts.jsonl
```

ARC-AGI-2 example rollouts:
```bash
ng_collect_rollouts +agent_name=arc_agi_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/example_2.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/example_2_rollouts.jsonl
```