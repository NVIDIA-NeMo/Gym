(training-nemotron-super)=
# Nemotron 3 Super Recipe

```{note}
This page is a stub. Content is being developed.
```

Train models using the Nemotron 3 Super recipe for maximum performance on multi-node clusters.

:::{card}

**Goal**: Train a production-quality model using the Nemotron 3 Super recipe on a multi-node cluster.

^^^

**In this tutorial, you will**:

1. Configure a multi-node cluster for distributed training
2. Set up the Nemotron 3 Super training configuration
3. Launch and monitor a production training run
4. Evaluate trained model performance

:::

---

## Before You Begin

Make sure you have these prerequisites ready:

- ✅ **Hardware**: 8+ nodes with 8× NVIDIA GPUs (80GB+ each, H100 recommended)
- ✅ **Software**: NeMo RL installed, Slurm cluster access
- ✅ **Storage**: 500 GB+ free disk space on a shared filesystem
- ✅ **Completed**: {doc}`Single-node training <../tutorials/nemo-rl-grpo/single-node-training>` tutorial recommended

**Optional accounts**:

- **Weights & Biases (W&B)**: For experiment tracking ([sign up](https://wandb.ai/signup))
- **HuggingFace**: For downloading models ([create token](https://huggingface.co/settings/tokens))

**Total time estimate**: ~6-12 hours (depending on cluster size and configuration)

---

## 1. Cluster Setup

**Estimated time**: ~30 minutes

### Verify Cluster Resources

<!-- TODO: Add cluster verification commands -->

**✅ Success Check**: All nodes report expected GPU count and connectivity.

### Configure Shared Storage

<!-- TODO: Document shared storage setup -->

**✅ Success Check**: All nodes can read/write to the shared filesystem.

---

## 2. Configuration

**Estimated time**: ~15 minutes

### Base Configuration

<!-- TODO: Add Nemotron 3 Super configuration file -->

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | TBD | Global batch size across all nodes |
| `learning_rate` | TBD | Peak learning rate |
| `num_steps` | TBD | Total training steps |
| `warmup_steps` | TBD | Learning rate warmup steps |

<!-- TODO: Document key hyperparameters -->

---

## 3. Launch Training

**Estimated time**: ~4-8 hours

### Submit Training Job

```bash
# TODO: Add training command
```

**✅ Success Check**: Training starts across all nodes. Check W&B for metrics or verify logs show progress.

### Monitor Progress

<!-- TODO: Document monitoring approach -->

---

## 4. Expected Results

After training completes, you should see improvements in these metrics:

| Metric | Baseline | After Training |
|--------|----------|----------------|
| Task completion | TBD | TBD |
| Tool call accuracy | TBD | TBD |

---

## Troubleshooting

### Common Issues

<!-- TODO: Add common issues and solutions -->

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM errors | Batch size too large | Reduce `batch_size` or enable gradient checkpointing |
| Slow training | Network bottleneck | Verify NCCL configuration |
| Node failures | Hardware issues | Check node health, restart failed nodes |

---

## What's Next?

After completing this tutorial, explore these options:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use Other Training Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse available resource servers on GitHub to find other training environments.
+++
{bdg-secondary}`github` {bdg-secondary}`resource-servers`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: ../tutorials/creating-resource-server
:link-type: doc

Create your own resource server with custom tools and verification logic.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`custom-tools`
:::

::::
