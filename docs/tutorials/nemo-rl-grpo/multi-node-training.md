(training-nemo-rl-grpo-multi-node-training)=

# Multi-Node Training

:::{important}
**Do the {doc}`Single Node Training <single-node-training>` first. Do not skip it.** The single-node setup validates that your environment is configured correctly before attempting multi-node training.
:::

For production training, scale to multiple nodes by changing `cluster.num_nodes`. This example uses **batch mode** (the `COMMAND` variable specifies what to run automatically when the job starts).

> **Note**: Run this command from the **Slurm login/head node**, not from inside the interactive container through Step 1. This submits a new batch job that will run independently.

Run this command to launch the training job! This uses the same configuration as the single node setup, just with a larger batch size for actual training purposes.

```bash
cd /path/to/nemo/rl

# Submit multi-node job
# Set these environment variables before running:
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   EXP_NAME: Experiment name
#   NUM_ACTOR_NODES: Number of GPU nodes to use (2, 4, 8, etc.)
#   CONTAINER_IMAGE_PATH: The container to use.
#   SLURM_ACCOUNT: Slurm account
#   SLURM_PARTITION: Slurm partition
WANDB_API_KEY={your W&B API key} \
EXP_NAME=nemo_gym_grpo/nemotron_nano_v2_9b/2nodes/workplace_assistant_001 \
NUM_ACTOR_NODES=2 \
REPO_LOCATION=$PWD \
CONTAINER_IMAGE_PATH=nvcr.io/nvidia/nemo-rl:v0.4.0 \
SLURM_ACCOUNT={your Slurm account} \
SLURM_PARTITION={your Slurm partition} \
    examples/nemo_gym/launch_nemo_gym_multinode_training.sh \
    --config=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

:::{tip}
If you are using enroot following the steps in the {doc}`Setup <setup>` doc and downloaded the container locally, please instead use the local container filepath:
```bash
CONTAINER_IMAGE_PATH=$PWD/../nvcr.io/nvidia/nemo-rl:v0.4.0 \
```
:::


## Expected Results

Monitor these metrics in W&B to track progress:
1. `train:reward_mean`: The average reward of your model on this training environment. The reward may be noisy, but it should go up.
2. `val:accuracy`: The validation performance of your model on this training environment. This should go up steadily.

The best checkpoint (highest `val:accuracy`) is retained based on `checkpointing.keep_top_k: 3`. You can find checkpoints at the following path:
```bash
ls results/$EXP_NAME
```


Training is successful when:
- Reward mean increases consistently over steps.
- Validation accuracy consistently improves.
- No OOM (Out of Memory) errors.
- Checkpoints are saved at specified intervals.

---

## Measuring Real-World Improvement

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement, evaluate the Nemotron Nano v2 9B model on BFCL v3 before and after training, and compare the results. You should observe measurable improvement in tool-calling accuracy.

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. Refer to the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

