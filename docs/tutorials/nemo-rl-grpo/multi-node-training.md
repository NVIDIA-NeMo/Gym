(training-nemo-rl-grpo-multi-node-training)=

# Multi-Node Training

Your single-node test run confirmed that the environment, model, and training loop all work correctly. Now you can scale to multiple nodes for production training, where the full power of distributed computing accelerates your GRPO optimization.

:::{card}

**Goal**: Scale GRPO training to multiple nodes for production training.

^^^

**In this section, you will**:

1. Launch a multi-node training job using Slurm batch mode
2. Monitor training metrics in Weights & Biases

:::

:::{button-ref} training-nemo-rl-grpo-single-node-training
:color: secondary
:outline:
:ref-type: ref

← Previous: Single Node Training
:::

---

## Before You Begin

:::{important}
**Complete the {doc}`Single Node Training <single-node-training>` first. Do not skip it.** The single-node setup validates that your environment is configured correctly before attempting multi-node training.
:::

Make sure you have:

- ✅ Successfully completed 3 training steps on a single node
- ✅ Access to the Slurm login/head node (not inside the interactive container)
- ✅ Weights & Biases API key for experiment tracking

---

## 1. Launch Multi-Node Training

**Estimated time**: Several hours (depending on configuration)

For production training, scale to multiple nodes by changing `cluster.num_nodes`. This example uses **batch mode**, where the `COMMAND` variable specifies what to run automatically when the job starts.

:::{note}
Run this command from the **Slurm login/head node**, not from inside the interactive container. This submits a new batch job that runs independently.
:::

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
CONTAINER_IMAGE_PATH=nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano \
SLURM_ACCOUNT={your Slurm account} \
SLURM_PARTITION={your Slurm partition} \
    examples/nemo_gym/launch_nemo_gym_multinode_training.sh \
    --config=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml \
    ++policy.generation.vllm_cfg.tool_parser_plugin=$(find $PWD/.cache -name nemotron_toolcall_parser_no_streaming.py) \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

:::{tip}
If you are using enroot following the steps in the {doc}`Setup <setup>` doc and downloaded the container locally, use the local container filepath instead:

```bash
CONTAINER_IMAGE_PATH=$PWD/../nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano \
```
:::

**✅ Success Check**: The Slurm job is submitted and begins running on multiple nodes.

---

## 2. Monitor Training Progress

Monitor these metrics in W&B to track progress:

| Metric | Description |
|--------|-------------|
| `train:reward_mean` | The average reward of your model on this training environment. The reward may be noisy, but it should go up. |
| `val:accuracy` | The validation performance of your model on this training environment. This should go up steadily. |

The best checkpoint (highest `val:accuracy`) is retained based on `checkpointing.keep_top_k: 3`. You can find checkpoints at the following path:

```bash
ls results/$EXP_NAME
```

**✅ Success Check**: Training is successful when:

- Reward mean increases consistently over steps
- Validation accuracy consistently improves
- No OOM (Out of Memory) errors occur
- Checkpoints are saved at specified intervals

---

## 3. Measure Real-World Improvement

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement, evaluate the Nemotron Nano v2 9B model on BFCL v3 before and after training, and compare the results. You should observe measurable improvement in tool-calling accuracy.

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. Refer to the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

**✅ Success Check**: BFCL v3 scores improve after training compared to the baseline model.

---

## What's Next?

Congratulations! You've trained Nemotron Nano 9B v2 for multi-step tool calling using GRPO on the Workplace Assistant environment.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use Other Training Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse available resource servers on GitHub to find other training environments.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: ../creating-resource-server
:link-type: doc

Create your own resource server with custom tools and verification logic.
:::

::::