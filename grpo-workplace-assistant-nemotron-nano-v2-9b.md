# GRPO Training with NeMo RL: Multi-step tool calling  on Nemotron Nano v2 9B

## Running Training

### Single Node Training (interactive mode)

**Estimated Time:** ~2-4 hours

Run these commands **from inside the container** after attaching via the interactive session from Step 1:

```bash
# Clean up any existing Ray/vLLM processes
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"

# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/nemotron_nano_v2_9b/workplace_assistant_001"

# Configuration file path
CONFIG_PATH=examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Set these environment variables before running:
#   HF_TOKEN: Your Hugging Face token for model downloads
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   TORCH_CUDA_ARCH_LIST: CUDA architectures compute capability
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

#   NRL_FORCE_REBUILD_VENVS: Set to true on first run to rebuild venvs
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=.cache/ \
HF_TOKEN="your_hugging_face_token" \
WANDB_API_KEY="your_wandb_api_key" \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="${USER}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME
```

### Multi-Node Training

For production training, scale to multiple nodes by changing `cluster.num_nodes`. This example uses **batch mode** (the `COMMAND` variable specifies what to run automatically when the job starts).

> **Note**: Run this command from the **Slurm login/head node**, not from inside the interactive container from Step 1. This submits a new batch job that will run independently.

```bash
# Set experiment name
EXP_NAME="penguin_grpo/nemotron_nano_v2_9b/2nodes/workplace_assistant_001"
CONFIG_PATH=examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Submit multi-node job
# Set these environment variables before running:
#   HF_TOKEN: Your Hugging Face token for model downloads
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   NUM_NODES: Number of GPU nodes to use (2, 4, 8, etc.)
NUM_NODES=2
COMMAND="TORCH_CUDA_ARCH_LIST='9.0 10.0' HF_HOME=.cache/ HF_TOKEN='your_hf_token' WANDB_API_KEY='your_wandb_api_key' uv run python examples/penguin/run_grpo_penguin.py --config=$CONFIG_PATH cluster.num_nodes=$NUM_NODES logger.wandb.project=${USER}-nemo-gym-rl-integration logger.wandb.name=$EXP_NAME logger.log_dir=results/$EXP_NAME checkpointing.checkpoint_dir=results/$EXP_NAME" \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.4.0 \
MOUNTS="/shared/filesystem:/shared/filesystem" \
sbatch \
    --nodes=$NUM_NODES \
    --time=4:0:0 \
    --job-name=grpo-workplace-assistant \
    --gres=gpu:8 \
    ray.sub
```

---

## Expected Results

### Training Metrics

Monitor these metrics in W&B to track progress:

| Metric | Initial | After 1 Epoch | Description |
|--------|---------|---------------|-------------|
| `train:reward_mean` | ~0.1-0.2 | ~0.5-0.7 | Average reward per batch |
| `val:accuracy` | ~0.15 | ~0.5-0.6 | Validation task completion rate |
| `train:loss` | ~0.5 | ~0.2-0.3 | GRPO policy loss |

### Checkpoint Outputs

Checkpoints are saved to:
```
results/<EXP_NAME>/
├── step_6/
├── step_12/
├── step_18/
└── ...
```

The best checkpoint (highest `val:accuracy`) is retained based on `checkpointing.keep_top_k: 3`.

### Success Criteria

Training is successful when:
- Reward mean increases consistently over steps
- Validation accuracy improves from baseline (~15%) to 50%+
- No OOM (Out of Memory) errors
- Checkpoints are saved at specified intervals

### Validation Reward Plot

<!-- TODO: Add validation reward plot showing improvement over training steps -->
![Validation Reward Plot](images/val_reward_placeholder.png)
*Expected: Validation reward increasing from ~0.15 to ~0.5+ over the course of training.*

### Measuring Real-World Improvement

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement, evaluate the Nemotron Nano v2 9B model on BFCL v3 before and after training and compare.  You should observe measurable improvement in tool-calling accuracy

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. See the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| HuggingFace rate limits | Specify your HF API token and/or add `HF_HUB_OFFLINE=1` after the initial download |
| vLLM process not shutting down | Run `pkill -f VllmAsyncGenerationWorker` before training |
| Ray cluster issues | Run `ray stop --force` before training |
| CUDA OOM | Increase `tensor_parallel_size`, lower batch sizes |
| Slow initial startup | Set `NRL_FORCE_REBUILD_VENVS=true` on first run only; if `uv` gets rate limited, set this back to `false` |

### Log Locations

```
logs/grpo-workplace-assistant-nemotron-nano-v2-9b/  # Training logs
results/<EXP_NAME>/                                  # Checkpoints and metrics
.cache/                                              # HuggingFace model cache
```

---

## Next Steps

After completing this tutorial, explore:

1. **Scale Up**: Try multi-node training for faster convergence and larger batch sizes
2. **Hyperparameter Tuning**: Adjust learning rate, number of generations, or reward normalization 
3. **Deploy Your Agent**: Export the trained checkpoint and deploy it with vLLM or NVIDIA NIM to build a production workplace assistant that integrates with real calendar, email, and file management APIs

### Related Tutorials

- [RL Training with NeMo RL](./rl-training-with-nemo-rl.md) - General RL training guide
- [GRPO Loss Configuration](../../docs/guides/grpo.md) - Advanced loss function customization
- [Sequence Packing](../../docs/design-docs/sequence-packing-and-dynamic-batching.md) - Optimize training throughput

---

## References

- **NeMo RL Repository**: [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)
- **NeMo Gym Repository**: [github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)

---

## Appendix: Full Configuration Reference

The complete training configuration is available at:

[`examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml)
