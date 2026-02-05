# ARC-AGI resources server

Abstraction and Reasoning Corpus for Artificial General Intelligence ([ARC-AGI](https://github.com/fchollet/ARC-AGI/)) is a benchmark with training and evaluation data to test general reasoning. It consists of grid-based puzzles, providing a few input output pairs and the system must infer the underlying abstract transformation rule and apply it to a new input. "ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence."

### Launch local vllm server
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

### Set env.yaml in `Gym/`: 
```
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### Create datasets: 
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

### Install Gym:
```
cd Gym/ 
uv venv 
source .venv/bin/activate
uv sync 
```

### Start ARC-AGI environment (we can reuse the same one for ARC-AGI-1 and 2):
```bash
ng_run "+config_paths=[resources_servers/arc_agi/configs/arc_agi.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```


### Collect rollouts:

ARC-AGI-1 example rollouts:
```bash
ng_collect_rollouts +agent_name=arc_agi_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/example_1.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/example_1_rollouts.jsonl
```

ARC-AGI-2 example rollouts:
```bash
ng_collect_rollouts +agent_name=arc_agi_simple_agent +input_jsonl_fpath=resources_servers/arc_agi/data/example_2.jsonl +output_jsonl_fpath=resources_servers/arc_agi/data/example_2_rollouts.jsonl
```

For training, see the [docs](https://docs.nvidia.com/nemo/gym/latest/tutorials/nemo-rl-grpo/index.html).