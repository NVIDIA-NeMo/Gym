(training-nemo-rl-grpo-single-node-training)=

# Single Node Training

TODO STILL

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



### Single node training

Now you're ready to launch a single-node training run to verify your setup works correctly before scaling to multi-node training.

**Prerequisites for this step**:

1. Completed the `ng_prepare_data` command above (data should be in `data/bytedtsinghua_dapo17k/`)
2. W&B API key (see [Prerequisites](#prerequisites) section above)
3. HuggingFace token configured (see setup steps above)


```bash
# Run example training config for single node
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/dapo17k_bytedtsinghua_test_001"
CONFIG_PATH=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml
HF_HOME=.cache/ \
WANDB_API_KEY={your W&B API key} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="{your username}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &
```

