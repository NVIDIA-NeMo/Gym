(training-nemo-rl-grpo-single-node-training)=

# Single Node Training

Now that you've completed the {doc}`Setup Instructions <setup>`, you're ready to launch a single-node training run!

**Estimated Time:** ~2-4 hours

Run these commands **from inside the container**.

:::{tip}
If you are performing these steps in a new container with existing code, just run these commands for the setup!
```bash
source /opt/nemo_rl_venv/bin/activate
uv sync --group={build,docs,dev,test} --extra nemo_gym
uv run nemo_rl/utils/prefetch_venvs.py
```
:::

Download NVIDIA [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
```bash
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
    hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2
```

Clean up any existing or leftover Ray/vLLM processes
```bash
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
```

```bash
# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/nemo_gym_grpo/nemotron_nano_v2_9b/workplace_assistant_001"

# Configuration file path
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Set these environment variables before running:
#   HF_TOKEN: Your Hugging Face token for model downloads
#   WANDB_API_KEY: Your Weights & Biases API key for logging
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
WANDB_API_KEY={your W&B API key} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="${USER}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    ++policy.generation.vllm_cfg.tool_parser_plugin=$(find .cache -name nemotron_toolcall_parser_no_streaming.py) \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    checkpointing.checkpoint_dir=results/$EXP_NAME
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

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement, evaluate the Nemotron Nano v2 9B model on BFCL v3 before and after training, and compare the results. You should observe measurable improvement in tool-calling accuracy.

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. Refer to the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

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
