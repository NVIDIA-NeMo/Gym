(training-nemo-rl-grpo-single-node-training)=

# Single Node Training

This section guides you through running a single-node GRPO training job.

:::{card}

**Goal**: Run a single-node GRPO training job and validate that training is progressing correctly.

^^^

**In this section, you will**:

1. Clean up existing processes
2. Launch single-node training
3. Monitor training metrics
4. Verify training success

:::

:::{button-ref} training-nemo-rl-grpo-setup
:color: secondary
:outline:
:ref-type: ref

← Previous: Setup
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`Setup <setup>`
- ✅ Inside the container on a GPU node
- ✅ (Optional) W&B API key for logging

**Estimated time**: ~2-4 hours

---

## 1. Clean Up Existing Processes

Before launching training, clean up any existing Ray or vLLM processes:

```bash
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
```

**✅ Success Check**: No errors (warnings about no processes found are fine).

---

## 2. Launch Training

Run these commands from inside the container:

```bash
# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/nemo_gym_grpo/nemotron_nano_v2_9b/workplace_assistant_001"

# Configuration file path
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Replace {your HF token} and {your W&B API key} with your actual values
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=.cache/ \
HF_TOKEN={your HF token} \
WANDB_API_KEY={your W&B API key} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="${USER}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &
```

**✅ Success Check**: Training starts and begins logging metrics.

---

## 3. Monitor Training Metrics

Track these metrics in W&B to monitor progress:

| Metric | Initial | After 1 Epoch | Description |
|--------|---------|---------------|-------------|
| `train:reward_mean` | ~0.1-0.2 | ~0.5-0.7 | Average reward per batch |
| `val:accuracy` | ~0.15 | ~0.5-0.6 | Validation task completion rate |
| `train:loss` | ~0.5 | ~0.2-0.3 | GRPO policy loss |

---

## 4. Verify Checkpoints

Checkpoints are saved to:

```
results/<EXP_NAME>/
├── step_6/
├── step_12/
├── step_18/
└── ...
```

The best checkpoint (highest `val:accuracy`) is retained based on `checkpointing.keep_top_k: 3`.

---

## Success Criteria

Training is successful when:

- ✅ Reward mean increases consistently over steps
- ✅ Validation accuracy improves from baseline (~15%) to 50%+
- ✅ No OOM (Out of Memory) errors
- ✅ Checkpoints are saved at specified intervals

---

## Validation Reward Plot

<!-- TODO: Add validation reward plot showing improvement over training steps -->

*Expected: Validation reward increasing from ~0.15 to ~0.5+ over the course of training.*

---

## Measuring Real-World Improvement

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement:

1. Evaluate Nemotron Nano v2 9B on BFCL v3 **before** training
2. Evaluate the trained model on BFCL v3 **after** training
3. Compare the results

You should observe measurable improvement in tool-calling accuracy.

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. Refer to the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HuggingFace rate limits | Specify your HF API token and/or add `HF_HUB_OFFLINE=1` after the initial download |
| vLLM process not shutting down | Run `pkill -f VllmAsyncGenerationWorker` before training |
| Ray cluster issues | Run `ray stop --force` before training |
| CUDA OOM | Increase `tensor_parallel_size`, lower batch sizes |
| Slow initial startup | Set `NRL_FORCE_REBUILD_VENVS=true` on first run only; set back to `false` if rate limited |

---

:::{button-ref} training-nemo-rl-grpo-multi-node-training
:color: primary
:ref-type: ref

Next: Multi-Node Training →
:::
