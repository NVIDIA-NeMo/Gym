(training-nemotron-nano)=
# Nemotron Nano Recipe

Quick-reference recipe for training models using the Nemotron Nano model family with NeMo RL and NeMo Gym.

**Nemotron Nano** is a family of efficient language models from NVIDIA designed for tool calling, reasoning, and general-purpose tasks. These models use hybrid architectures (Transformer + Mamba) or Mixture-of-Experts (MoE) for efficient training and inference.

**Use this page to**: select a model variant, check hardware requirements, and copy working launch commands.

:::{tip}
**New to NeMo Gym training?** Start with the {doc}`NeMo RL GRPO Tutorial <../tutorials/nemo-rl-grpo/index>` for a complete walkthrough before using this recipe.
:::

---

## Nemotron Nano Model Family

The Nemotron Nano family includes multiple model variants optimized for different use cases:

| Model | Parameters | Architecture | HuggingFace ID | Best For |
|-------|-----------|--------------|----------------|----------|
| Nemotron Nano v2 9B | 9B | Hybrid (Transformer + Mamba) | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | Single-node training, tool calling |
| Nemotron Nano v2 12B | 12B | Hybrid (Transformer + Mamba) | `nvidia/NVIDIA-Nemotron-Nano-12B-v2` | Multi-node training, higher capacity |
| Nemotron 3 Nano 30B | 30B (3B active) | MoE | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Reasoning tasks, inference |

---

## Prerequisites

Hardware requirements vary by model variant (approximate):

### Nemotron Nano v2 9B

- **GPUs**: 8× NVIDIA GPUs (24GB+ VRAM, H100 recommended)
- **Nodes**: 1-2 nodes
- **Storage**: ~20 GB for model weights

### Nemotron Nano v2 12B

- **GPUs**: 8× NVIDIA GPUs (40GB+ VRAM)
- **Nodes**: 1-2 nodes
- **Storage**: ~30 GB for model weights

### Nemotron 3 Nano 30B

- **GPUs**: 8× NVIDIA GPUs (80GB VRAM, H100 required)
- **Nodes**: 1+ nodes
- **Storage**: ~60 GB for model weights

**Common Requirements**:

- NeMo RL v0.4.0+ installed ([setup instructions](../tutorials/nemo-rl-grpo/setup))
- HuggingFace token for model download
- (Optional) Weights & Biases API key for experiment tracking

:::{warning}
Never commit your HuggingFace token to version control. Use environment variables or a local `env.yaml` file that is excluded from Git.
:::

---

## Quick Start

### Nemotron Nano v2 9B (Recommended)

This is the recommended starting point for tool-calling training with GRPO (Group Relative Policy Optimization). For detailed step-by-step instructions, see the {doc}`NeMo RL GRPO Tutorial <../tutorials/nemo-rl-grpo/index>`.

**1. Download the model**:

```bash
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
    hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2
```

**2. Configure the chat template**:

```bash
tokenizer_config_path=$(find $PWD/.cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2 -name tokenizer_config.json)
sed -i 's/enable_thinking=true/enable_thinking=false/g' $tokenizer_config_path
```

**3. Launch training**:

```bash
EXP_NAME="$(date +%Y%m%d)/nemotron_nano_v2_9b/workplace_assistant"
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
mkdir -p results/$EXP_NAME

HF_HOME=$PWD/.cache/ \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$CONFIG_PATH \
    ++logger.log_dir=results/$EXP_NAME \
    ++policy.generation.vllm_cfg.tool_parser_plugin=$(find $PWD/.cache -name nemotron_toolcall_parser_no_streaming.py) \
    ++grpo.max_num_steps=3
```

### Nemotron Nano v2 12B

Use the 12B variant for higher model capacity with Megatron backend.

**1. Download the model**:

```bash
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
    hf download nvidia/NVIDIA-Nemotron-Nano-12B-v2
```

**2. Launch training**:

```bash
HF_HOME=$PWD/.cache/ \
uv run python examples/run_grpo_math.py \
    --config=examples/configs/recipes/llm/grpo-nano-v2-12b-1n8g-megatron.yaml \
    ++logger.log_dir=results/nano-v2-12b
```

For multi-node training with FSDP:

```bash
HF_HOME=$PWD/.cache/ \
uv run python examples/run_grpo_math.py \
    --config=examples/configs/recipes/llm/grpo-nano-v2-12b-2n8g-fsdp2tp1.yaml \
    ++cluster.num_nodes=2 \
    ++logger.log_dir=results/nano-v2-12b-multinode
```

---

## Configuration Reference

### Nemotron Nano v2 9B (Workplace Assistant)

Key parameters from `examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`:

| Category | Parameter | Value | Description |
|----------|-----------|-------|-------------|
| **Model** | `model_name` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | HuggingFace model ID |
| | `max_total_sequence_length` | 8192 | Maximum context length |
| | `precision` | `bfloat16` | Training precision |
| **GRPO** | `num_prompts_per_step` | 64 | Prompts sampled per training step |
| | `num_generations_per_prompt` | 16 | Rollouts generated per prompt |
| | `use_leave_one_out_baseline` | `true` | Variance reduction technique |
| | `normalize_rewards` | `true` | Normalize rewards across batch |
| **Optimizer** | `lr` | `5.0e-6` | Peak learning rate |
| | `min_lr` | `5.0e-7` | Minimum learning rate |
| | `weight_decay` | `0.01` | Weight decay |
| | `adam_beta1` / `adam_beta2` | `0.9` / `0.999` | Adam hyperparameters |
| **Parallelism** | `tensor_model_parallel_size` | 2 | Tensor parallelism degree |
| | `activation_checkpointing` | `true` | Memory optimization |
| **Generation** | `backend` | `vllm` | Generation backend |
| | `temperature` | `1.0` | Sampling temperature |
| | `tool_parser` | `nemotron_json` | Tool call parser for vLLM |

### Nemotron Nano v2 12B (Megatron)

Key parameters from `examples/configs/recipes/llm/grpo-nano-v2-12b-1n8g-megatron.yaml`:

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | `model_name` | `nvidia/NVIDIA-Nemotron-Nano-12B-v2` |
| **Parallelism** | `tensor_model_parallel_size` | 8 |
| | `megatron_cfg.enabled` | `true` |
| **Data** | `max_input_seq_length` | 512 |
| **Generation** | `max_model_len` | 512 |

### Nemotron Nano v2 12B (FSDP Multi-Node)

Key parameters from `examples/configs/recipes/llm/grpo-nano-v2-12b-2n8g-fsdp2tp1.yaml`:

| Category | Parameter | Value |
|----------|-----------|-------|
| **Cluster** | `num_nodes` | 2 |
| | `gpus_per_node` | 8 |
| **DTensor** | `cpu_offload` | `true` |
| | `activation_checkpointing` | `true` |
| **Batching** | `dynamic_batching.enabled` | `true` |

---

## vLLM Configuration for Nemotron Nano

Nemotron Nano v2 models require specific vLLM settings for tool calling:

```yaml
vllm_cfg:
  tool_parser_plugin: /path/to/nemotron_toolcall_parser_no_streaming.py
  http_server_serving_chat_kwargs:
    enable_auto_tools: true
    tool_parser: nemotron_json
vllm_kwargs:
  mamba_ssm_cache_dtype: "float32"  # Required for Nemotron Nano v2
```

:::{important}
The Mamba cache must use `float32` precision for Nemotron Nano v2 models. Using lower precision may cause numerical instability.
:::

---

## Troubleshooting

### Chat Template Issues

If tool calling fails, verify the chat template was modified correctly:

```bash
# Check that thinking mode is disabled
grep "enable_thinking" $tokenizer_config_path
```

The output should show `enable_thinking=false`.

### Out of Memory

For OOM errors, try:

1. Enable CPU offload: `++policy.dtensor_cfg.cpu_offload=true`
2. Enable activation checkpointing: `++policy.megatron_cfg.activation_checkpointing=true`
3. Reduce batch size: `++policy.train_micro_batch_size=1`

### vLLM Tool Parser Not Found

Ensure the tool parser plugin path is correct:

```bash
find $PWD/.cache -name "nemotron_toolcall_parser_no_streaming.py"
```

If not found, re-download the model with `hf download`.

---

## Related Resources

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Complete GRPO Tutorial
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc

Step-by-step guide covering setup, single-node, and multi-node training with Nemotron Nano 9B v2.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` NeMo RL Configuration Reference
:link: ../tutorials/nemo-rl-grpo/nemo-rl-configuration
:link-type: doc

Detailed explanation of all GRPO and model hyperparameters.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Workplace Assistant Environment
:link: ../tutorials/nemo-rl-grpo/about-workplace-assistant
:link-type: doc

Learn about the tool-calling training environment used with Nemotron Nano.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Multi-Node Training
:link: ../tutorials/nemo-rl-grpo/multi-node-training
:link-type: doc

Scale training to multiple nodes for production workloads.
:::

::::
