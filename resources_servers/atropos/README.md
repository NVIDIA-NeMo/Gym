# Atropos Integration for NeMo Gym

Integrates [Atropos](https://github.com/NousResearch/atropos)

## Quickstart

```bash
# Start vLLM
HF_HOME=.cache/ \
HOME=. \
vllm serve \
    Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240

# Prep data
cd resources_servers/atropos
python scripts/generate_task_indices.py --num_samples 5 --output data/gsm8k_sample.jsonl

# Set env.yaml
cat > env.yaml << 'EOF'
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-Nano-9B-v2
EOF

# Install Nemo Gym, then
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/atropos/configs/gsm8k_atropos.yaml"
ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=gsm8k_atropos_agent \
    +input_jsonl_fpath=resources_servers/atropos/data/gsm8k_sample.jsonl \
    +output_jsonl_fpath=results/gsm8k_rollouts.jsonl \
    +limit=5
```

## Tool Calling

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/atropos/configs/tool_calling.yaml"
ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=tool_calling_atropos_agent \
    +input_jsonl_fpath=resources_servers/atropos/data/tool_calling_sample.jsonl \
    +output_jsonl_fpath=results/tool_calling_rollouts.jsonl \
    +limit=5
```

## Letter Counting

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/atropos/configs/letter_counting.yaml"
ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=letter_counting_atropos_agent \
    +input_jsonl_fpath=resources_servers/atropos/data/letter_counting_train.jsonl \
    +output_jsonl_fpath=results/letter_counting_rollouts.jsonl \
    +limit=5
```

## KernelBench

KernelBench has 270 problems across 4 levels:
- Level 1: Single-kernel operators (100)
- Level 2: Fusion patterns (100)
- Level 3: Model architectures (50)
- Level 4: HF models (20)

```bash
# Hack: using gpu 0 for kernels by default (KERNELBENCH_DEVICE in config), and launching vllm on 1-N, to avoid memory conflict

cd /path/to/Gym
git clone https://github.com/ScalingIntelligence/KernelBench.git

config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/atropos/configs/kernelbench.yaml"
ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=kernelbench_atropos_agent \
    +input_jsonl_fpath=resources_servers/atropos/data/kernelbench_train.jsonl \
    +output_jsonl_fpath=results/kernelbench_rollouts.jsonl \
    +limit=5
```
