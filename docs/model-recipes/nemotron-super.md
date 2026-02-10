(training-nemotron-super)=
# Production-Scale Training: Nemotron Super 49B

```{warning}
**Experimental Configuration**: This tutorial uses configurations marked `.disabled` in NeMo RL, indicating they are not yet fully validated. Known issues include [GitHub #1571](https://github.com/NVIDIA-NeMo/RL/issues/1571).

**Recommended alternative**: The {doc}`NeMo RL GRPO tutorial <../tutorials/nemo-rl-grpo/index>` provides a validated single-node workflow with Nemotron Nano 9B.
```

Train large language models at production scale using the Llama-3.3-Nemotron-Super-49B model on multi-node clusters with NeMo RL.

:::{card}

**Goal**: Configure and launch production-scale training on a multi-node cluster.

^^^

**In this tutorial, you will**:

1. Understand hardware requirements and cost implications
2. Configure the Nemotron Super 49B training environment
3. Launch SFT or GRPO training across multiple nodes
4. Monitor training and handle common failure scenarios

:::

:::{button-ref} ../tutorials/nemo-rl-grpo/index
:color: secondary
:outline:
:ref-type: doc

← Previous: NeMo RL GRPO Tutorial
:::

---

## Before You Begin

### When to Use This vs. Nano

| Factor | Nemotron Super 49B | Nemotron Nano 9B |
|--------|-------------------|------------------|
| **Parameters** | 49B | 9B |
| **GPU requirement** | 64-128× H100 (80GB) | 8× H100 (80GB) |
| **Estimated cloud cost** | $5,000-$15,000 per run | $200-$500 per run |
| **Training time** | 4-12 hours | 1-4 hours |
| **Validation status** | Experimental | ✅ Validated |
| **Best for** | Production deployment, research | Iteration, prototyping |

**Use Super 49B when**: You need maximum model quality and have the compute budget.
**Use Nano 9B when**: You want to iterate quickly or validate your approach first.

### Hardware Requirements

| Requirement | Specification |
|-------------|---------------|
| **Nodes** | 8+ nodes (SFT) or 16+ nodes (GRPO) |
| **GPUs per node** | 8× NVIDIA H100 (80GB) |
| **Total GPUs** | 64-128 GPUs |
| **Storage** | 500 GB+ on shared filesystem (Lustre, GPFS, NFS) |
| **Network** | NVLink + InfiniBand (200+ Gb/s recommended) |

:::{dropdown} What is heterogeneous architecture?
:icon: info

The Nemotron Super 49B uses a **heterogeneous transformer** where different layers have different configurations (varying attention patterns, MLP sizes). This is implemented via the `DeciLMForCausalLM` model class.

**Why it matters**: Standard tensor parallelism won't work out-of-the-box. You need a custom parallel plan that maps each layer correctly.

**Technical detail**: The model's `config.json` contains a `block_configs` field specifying per-layer architecture.
:::

### Software Requirements

- ✅ **NeMo RL**: [Installation guide](https://github.com/NVIDIA-NeMo/RL#installation)
- ✅ **Slurm**: Cluster job scheduler
- ✅ **[uv](https://docs.astral.sh/uv/)**: Fast Python package manager (used by NeMo RL)
- ✅ **Ray**: Distributed coordination (installed with NeMo RL)
- ✅ **vLLM**: Efficient generation (installed with NeMo RL)

:::{dropdown} New to Slurm?
:icon: light-bulb

Slurm is a cluster job scheduler. Key commands:
- `sinfo`: Show cluster status
- `srun`: Run interactive jobs
- `sbatch`: Submit batch jobs
- `scancel`: Cancel jobs
- `squeue -u $USER`: Show your queued jobs

[Slurm quickstart guide](https://slurm.schedmd.com/quickstart.html)
:::

### Required Accounts

- **HuggingFace**: Model access requires a token ([create token](https://huggingface.co/settings/tokens))

### Optional Accounts

- **Weights & Biases (W&B)**: Experiment tracking ([sign up](https://wandb.ai/signup))

### Terminology Reference

| Term | Meaning |
|------|---------|
| **TP (Tensor Parallel)** | Splits model layers across GPUs within a node |
| **CP (Context Parallel)** | Splits sequence processing across GPUs for long contexts |
| **DP (Data Parallel)** | Each GPU processes different data batches |
| **FSDP2** | Fully Sharded Data Parallel v2 — shards optimizer/gradients across GPUs |
| **Activation checkpointing** | Trades compute for memory by recomputing activations during backward pass |

---

## 1. Cluster Setup

**Estimated time**: ~30 minutes

### Verify Cluster Resources

```bash
# Check available nodes
sinfo -N -l

# Verify GPU count per node (requires Slurm allocation)
srun --nodes=1 --gpus-per-node=8 --time=5:00 nvidia-smi -L

# Test multi-node connectivity
srun --nodes=8 --ntasks-per-node=1 --time=5:00 hostname
```

**✅ Success Check**: All 8 nodes report 8× H100 GPUs with 80GB memory each.

### Configure Environment

```bash
# Set up experiment directory (adjust path for your cluster)
export SHARED_FS=/lustre/scratch/$USER  # or your shared filesystem
export EXPERIMENT_DIR=$SHARED_FS/nemotron-super-$(date +%Y%m%d)
mkdir -p $EXPERIMENT_DIR

# Verify all nodes can access it
srun --nodes=8 --ntasks-per-node=1 touch $EXPERIMENT_DIR/node_\$SLURM_NODEID.txt
ls $EXPERIMENT_DIR/  # Should show node_0.txt through node_7.txt
rm $EXPERIMENT_DIR/node_*.txt  # Cleanup

# Set HuggingFace token (required for gated model)
export HF_TOKEN=hf_your_token_here  # Replace with your actual token
```

**✅ Success Check**: All nodes can read/write to the shared directory.

---

## 2. Configuration

**Estimated time**: ~15 minutes

### Model Information

| Property | Value |
|----------|-------|
| **Model name** | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` |
| **Parameters** | 49 billion |
| **Architecture** | Heterogeneous transformer (DeciLMForCausalLM) |
| **Max sequence length** | 32,768 tokens |
| **trust_remote_code** | Required (`True`) |

```{warning}
**Security note**: This model requires `trust_remote_code=True`, which executes code from the HuggingFace repository. Review the [model card](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) before proceeding.
```

### Key Hyperparameters

:::::{tab-set}

::::{tab-item} SFT Configuration

Based on `examples/configs/recipes/llm/sft-nemotron-super-49b-8n8g-fsdp2tp4cp8-tulu-v3.yaml.disabled`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train_global_batch_size` | 128 | Total batch across all nodes |
| `learning_rate` | 1.0e-05 | Peak learning rate |
| `max_num_steps` | 50 | Training iterations |
| `warmup_steps` | 10 | Linear warmup iterations |
| `max_total_sequence_length` | 32768 | Maximum context window |
| `tensor_parallel_size` | 4 | GPUs for tensor parallelism |
| `context_parallel_size` | 8 | GPUs for context parallelism |
| `activation_checkpointing` | true | Memory optimization |

**Dataset**: TULU v3 SFT Mixture (`tulu3_sft_mixture`)

**Cluster**: 8 nodes × 8 GPUs = **64 GPUs**

::::

::::{tab-item} GRPO Configuration

Based on `examples/configs/recipes/llm/grpo-helpsteer3-llama-3.3-nemotron-super-49b-v1.5-8n8g-fsdp2tp8cp4.yaml.disabled`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train_global_batch_size` | 64 | Total batch across all nodes |
| `learning_rate` | 3.0e-07 | Peak learning rate (lower for RL) |
| `max_num_steps` | 10 | Training iterations |
| `num_prompts_per_step` | 64 | Prompts per training step |
| `tensor_parallel_size` | 8 | GPUs for tensor parallelism |
| `context_parallel_size` | 4 | GPUs for context parallelism |
| `cpu_offload` | true | Optimizer CPU offloading |

**Dataset**: HelpSteer3

**Cluster**: 16 nodes × 8 GPUs = **128 GPUs**

::::

:::::

### Copy and Modify Configuration

The reference configurations are marked `.disabled`. Copy and modify them:

```bash
# Navigate to NeMo RL directory
cd /path/to/NeMo-RL

# Copy the SFT config
cp examples/configs/recipes/llm/sft-nemotron-super-49b-8n8g-fsdp2tp4cp8-tulu-v3.yaml.disabled \
   examples/configs/recipes/llm/my-nemotron-super-sft.yaml

# Or copy the GRPO config
cp examples/configs/recipes/llm/grpo-helpsteer3-llama-3.3-nemotron-super-49b-v1.5-8n8g-fsdp2tp8cp4.yaml.disabled \
   examples/configs/recipes/llm/my-nemotron-super-grpo.yaml
```

Edit the checkpoint directory in your config:

```yaml
# In your copied config file, update:
checkpointing:
  checkpoint_dir: /lustre/scratch/your_username/nemotron-super/checkpoints
```

---

## 3. Launch Training

**Estimated time**: 4-12 hours

### Complete Slurm Job Script

Create a job script `train_nemotron_super.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=nemotron-super-sft
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu  # Adjust for your cluster
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# === Environment Setup ===
export HF_TOKEN=hf_your_token_here  # Replace with your token
export HF_HOME=$SCRATCH/.cache/huggingface
export WANDB_API_KEY=your_wandb_key  # Optional: for W&B logging

# Ensure PYTHONPATH includes examples for custom parallel plan
export PYTHONPATH=$PWD:$PYTHONPATH

# === Configuration ===
CONFIG_PATH=examples/configs/recipes/llm/my-nemotron-super-sft.yaml
EXP_NAME="nemotron_super_sft_$(date +%Y%m%d_%H%M%S)"

# === Launch Training ===
srun uv run python examples/run_sft.py \
    --config=$CONFIG_PATH \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=$SCRATCH/logs/$EXP_NAME

echo "Training complete. Logs at: $SCRATCH/logs/$EXP_NAME"
```

Submit the job:

```bash
sbatch train_nemotron_super.sbatch
```

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f nemotron-super-sft-*.out

# Check GPU utilization (while job is running)
srun --jobid=YOUR_JOB_ID --overlap nvidia-smi
```

### Cancel and Cleanup

```bash
# Cancel a running job
scancel YOUR_JOB_ID

# Clean up checkpoints (WARNING: deletes data)
# rm -rf $SCRATCH/nemotron-super/checkpoints/*
```

**✅ Success Check**: Training starts across all nodes. Check logs for:
- `Training step 1/50` messages
- No OOM errors
- GPU utilization >80%

---

## 4. Expected Results

```{note}
These are estimates based on similar large-scale training runs. Your results will vary based on dataset, hardware, and hyperparameters. The experimental status of these configs means you may encounter issues.
```

### SFT Training (64 GPUs, 50 steps)

| Metric | Expected |
|--------|----------|
| Training loss | Start ~2.5, end ~1.5 |
| Validation loss | Should track training ±0.2 |
| Throughput | ~50,000-100,000 tokens/sec total |
| Time per step | ~2-5 minutes |
| Total time | ~2-4 hours |

### GRPO Training (128 GPUs, 10 steps)

| Metric | Expected |
|--------|----------|
| Mean reward | Increasing trend |
| KL divergence | <10 (bounded) |
| Policy loss | Decreasing trend |
| Time per step | ~10-20 minutes |
| Total time | ~2-4 hours |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM errors | Batch size too large | Add `++policy.dtensor_cfg.cpu_offload=true` |
| `ModuleNotFoundError: custom_parallel` | PYTHONPATH not set | Add `export PYTHONPATH=$PWD:$PYTHONPATH` |
| Model download fails | Missing HF token | Verify `HF_TOKEN` is set and valid |
| Slow training | Network bottleneck | Check `NCCL_DEBUG=INFO` output |
| Node failures mid-training | Hardware issues | Use `scontrol show node` to check health |

### Checkpoint Recovery

If training is interrupted:

```bash
# Find latest checkpoint
ls -la $SCRATCH/nemotron-super/checkpoints/

# Resume from checkpoint (add to your command)
# ++checkpointing.load_path=$SCRATCH/nemotron-super/checkpoints/step_40
```

### Verifying Model Compatibility

```bash
# Test that the model loads correctly
HF_TOKEN=hf_your_token python -c "
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained(
    'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5',
    trust_remote_code=True
)
print('✅ Model type:', config.model_type)
print('✅ Has block_configs:', hasattr(config, 'block_configs'))
print('✅ Architecture: DeciLMForCausalLM (heterogeneous)')
"
```

---

## Production Checklist

Before running production training:

- [ ] **Budget approved**: 64-128× H100 for 4-12 hours (~$5,000-$15,000 cloud cost)
- [ ] **Storage allocated**: 500GB+ for checkpoints
- [ ] **Tested on Nano first**: Validated approach with {doc}`Nano tutorial <../tutorials/nemo-rl-grpo/index>`
- [ ] **HF_TOKEN set**: Model access configured
- [ ] **W&B configured**: Experiment tracking (optional but recommended)
- [ ] **Checkpoint path writable**: All nodes can write to checkpoint directory
- [ ] **Job time limit adequate**: Set `--time` >= expected runtime + buffer

---

## What's Next?

After completing this tutorial, explore these options:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` NeMo RL GRPO Tutorial
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc

Complete, validated tutorial for single-node training with Nemotron Nano 9B.
+++
{bdg-primary}`recommended` {bdg-secondary}`validated`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: ../environment-tutorials/creating-training-environment
:link-type: doc

Create your own resource server with custom tools and verification logic.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`custom-tools`
:::

::::

---

## References

- **Model**: [nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) on HuggingFace
- **SFT Config**: `examples/configs/recipes/llm/sft-nemotron-super-49b-8n8g-fsdp2tp4cp8-tulu-v3.yaml.disabled`
- **GRPO Config**: `examples/configs/recipes/llm/grpo-helpsteer3-llama-3.3-nemotron-super-49b-v1.5-8n8g-fsdp2tp8cp4.yaml.disabled`
- **Custom Parallel Plan**: `examples/custom_parallel/llama_nemotron_super_49b_custom_plan.py`
- **Known Issues**: [GitHub #1571](https://github.com/NVIDIA-NeMo/RL/issues/1571) — Model name compatibility
