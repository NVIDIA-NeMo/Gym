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
Efficient small model training.
+++
{bdg-secondary}`nemotron` {bdg-secondary}`nano`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Nemotron 3 Super
:link: nemotron-super
:link-type: doc
High-performance training.
+++
{bdg-secondary}`nemotron` {bdg-secondary}`super`
:::

::::

## Framework Tutorials

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` NeMo RL with GRPO
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc
Multi-step tool calling with GRPO.
+++
{bdg-primary}`recommended` {bdg-secondary}`grpo`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth Training
:link: ../tutorials/unsloth-training
:link-type: doc
Fast, memory-efficient fine-tuning.
+++
{bdg-secondary}`unsloth` {bdg-secondary}`efficient`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` TRL Training
:link: trl
:link-type: doc
HuggingFace TRL integration.
+++
{bdg-secondary}`trl` {bdg-secondary}`huggingface`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` VeRL Training
:link: verl
:link-type: doc
VeRL framework integration.
+++
{bdg-secondary}`verl`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` NeMo Customizer
:link: nemo-customizer
:link-type: doc
NeMo Customizer integration.
+++
{bdg-secondary}`nemo-customizer`
:::

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Offline Training
:link: ../tutorials/offline-training-w-rollouts
:link-type: doc
SFT and DPO from rollouts.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::

## Choosing a Framework

- **Production training**: Use NeMo RL for multi-node GRPO training
- **Rapid prototyping**: Use Unsloth for fast single-GPU iteration
- **HuggingFace models**: Use TRL for seamless ecosystem integration
- **Offline training**: Use SFT/DPO when you have high-quality rollouts
