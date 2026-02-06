(training-tutorials-index)=
# Training Tutorials

Hands-on tutorials for training models with NeMo Gym across different frameworks and configurations.

## Training Frameworks

NeMo Gym integrates with various RL training frameworks:

| Framework | Algorithm | GPU Support | Best For |
|-----------|-----------|-------------|----------|
| [NeMo RL](../tutorials/nemo-rl-grpo/index) | GRPO | Multi-node | Production training |
| [Unsloth](../tutorials/unsloth-training) | Various | Single GPU | Fast iteration |
| [TRL](trl) | PPO, DPO | Multi-GPU | HuggingFace ecosystem |
| [VeRL](verl) | Various | Multi-node | Research |

## Recipe Tutorials

Pre-configured training recipes for specific models:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Nemotron 3 Nano
:link: nemotron-nano
:link-type: doc
Single-node training recipe for Nemotron Nano 9B using GRPO on Workplace Assistant tasks.
+++
{bdg-primary}`validated` {bdg-secondary}`1-4 hours` {bdg-secondary}`single-node`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Nemotron 3 Super
:link: nemotron-super
:link-type: doc
Multi-node production training for Nemotron Super 49B. Requires 64-128 H100 GPUs.
+++
{bdg-secondary}`experimental` {bdg-secondary}`5-12 hours` {bdg-secondary}`multi-node`
:::

::::

## Framework Tutorials

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` NeMo RL with GRPO
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc
Multi-page series: train Nemotron Nano 9B for multi-step tool calling on single and multi-node clusters.
+++
{bdg-primary}`recommended` {bdg-secondary}`3-5 hours` {bdg-secondary}`grpo`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth Training
:link: ../tutorials/unsloth-training
:link-type: doc
Fine-tune on single GPU using Colab. Supports math, structured outputs, and reasoning tasks.
+++
{bdg-secondary}`30 min` {bdg-secondary}`unsloth` {bdg-secondary}`single-gpu`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` TRL Training
:link: trl
:link-type: doc
PPO, DPO, and GRPO training with HuggingFace Transformers ecosystem integration.
+++
{bdg-secondary}`trl` {bdg-secondary}`huggingface`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` VeRL Training
:link: verl
:link-type: doc
Distributed RL training for research workflows with flexible algorithm support.
+++
{bdg-secondary}`verl` {bdg-secondary}`research`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` NeMo Customizer
:link: nemo-customizer
:link-type: doc
Enterprise training and deployment with NeMo Customizer managed service.
+++
{bdg-secondary}`nemo-customizer` {bdg-secondary}`enterprise`
:::

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Offline Training
:link: ../tutorials/offline-training-w-rollouts
:link-type: doc
Transform collected rollouts into SFT and DPO training datasets without online exploration.
+++
{bdg-secondary}`20 min` {bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::

## Choosing a Framework

- **Production training**: Use NeMo RL for multi-node GRPO training
- **Rapid prototyping**: Use Unsloth for fast single-GPU iteration
- **HuggingFace models**: Use TRL for seamless ecosystem integration
- **Offline training**: Use SFT/DPO when you have high-quality rollouts
