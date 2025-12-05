(tutorials-integrate-training-frameworks)=

# Integrate Training Frameworks

Connect NeMo Gym to popular training frameworks. Follow these tutorials to go from collected rollouts to a trained model checkpoint.

## Before You Start

All tutorials in this series assume you have:

- ✅ Completed [Collecting Rollouts](../get-started/rollout-collection.md)
- ✅ A rollouts file (`results/rollouts.jsonl`) with scored interactions
- ✅ Understanding of your training objective (SFT, DPO, or RL)

---

## Choose Your Framework

Select the training framework that best fits your infrastructure and requirements.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`hubot;1.5em;sd-mr-1` Train with TRL
:link: train-with-trl
:link-type: doc

Use Hugging Face's Transformer Reinforcement Learning library for SFT, DPO, or GRPO training.
+++
{bdg-primary}`recommended`
{bdg-secondary}`hugging-face`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: train-with-nemo-rl
:link-type: doc

Use NVIDIA's NeMo RL for distributed on-policy training with NeMo 2.0 models.
+++
{bdg-secondary}`nvidia`
{bdg-secondary}`multi-node`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Train with VeRL
:link: train-with-verl
:link-type: doc

Use VeRL's Ray-based distributed training with flexible backend support (vLLM, SGLang, HF).
+++
{bdg-secondary}`ray`
{bdg-secondary}`multi-backend`
:::

::::

---

## Background Reading

Want to understand how Gym integrates with training frameworks before diving in?

- {doc}`/about/concepts/training-integration-architecture` — Request lifecycle, multi-turn rollout flow, and token alignment challenges

---

```{toctree}
:maxdepth: 1
:hidden:

TRL <train-with-trl>
RL <train-with-nemo-rl>
VeRL <train-with-verl>
```

