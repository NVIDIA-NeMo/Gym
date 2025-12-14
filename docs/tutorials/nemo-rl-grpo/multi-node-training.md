(training-nemo-rl-grpo-multi-node-training)=

# Multi-Node Training

:::{important}
**Do the {doc}`Single Node Training <single-node-training>` first. Do not skip it.** The single-node setup validates that your environment is configured correctly before attempting multi-node training.
:::

For production training, scale to multiple nodes by changing `cluster.num_nodes`. This example uses **batch mode** (the `COMMAND` variable specifies what to run automatically when the job starts).

> **Note**: Run this command from the **Slurm login/head node**, not from inside the interactive container through Step 1. This submits a new batch job that will run independently.

We will write our Slurm job launch script and then run it.

## Submit script

Place this script (named, for example, `temp_nemo_gym_submit.sh`) in the root NeMo RL dir.

```bash
# ----- PARAMETERS -----
# WANDB_API_KEY, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION

# ----- CONSTANTS -----
CONTAINER_IMAGE_PATH=/path/to/nemo-rl/container

read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=$PWD/.cache/ \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    cluster.num_nodes=$NUM_ACTOR_NODES \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    checkpointing.checkpoint_dir=results/$EXP_NAME \
    $@
EOF

echo -e "Running command:\n$COMMAND"

# ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="/shared/filesystem:/shared/filesystem" \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
```


## Submit script run

Run this command to launch the training job! This uses the same configuration as the single node setup, just with a larger batch size for actual training purposes.

```bash
# Submit multi-node job
# Set these environment variables before running:
#   HF_TOKEN: Your Hugging Face token for model downloads
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   NUM_NODES: Number of GPU nodes to use (2, 4, 8, etc.)
HF_TOKEN={your HF token} \
WANDB_API_KEY={your W&B API key} \
EXP_NAME=nemo_gym_grpo/qwen3_4binstruct/8nodes/dapo17k_bytedtsinghua_nf_001 \
NUM_ACTOR_NODES=8 \
REPO_LOCATION={your NeMo RL dir}\
    ./temp_nemo_gym_submit.sh \
    --config=examples/nemo_gym/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

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

