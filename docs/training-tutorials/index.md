(training-tutorials-index)=
# Training Tutorials

We have hands-on tutorials with supported training frameworks to help you train with NeMo Gym environments. If you're interested in integrating another training framework, see the {doc}`Training Framework Integration Guide <../contribute/rl-framework-integration/index>`.

## RL (GRPO)

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` NeMo RL
:link: nemo-rl-grpo/index
:link-type: doc
Tutorial-series: GRPO training to improve multi-step tool calling on the Workplace Assistant environment, scaling from single-node to multi-node training.
+++
{bdg-secondary}`nemo rl` {bdg-secondary}`grpo` {bdg-secondary}`3-5 hours`
:::

:::{grid-item-card} {octicon}`link-external;1.5em;sd-mr-1` OpenRLHF
:link: https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_nemogym_executor.py
:link-type: url
Review the agent executor for using NeMo Gym environments with OpenRLHF.
+++
{bdg-secondary}`openrlhf`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` TRL
:link: trl
:link-type: doc
GRPO training on Workplace Assistant and Reasoning Gym environments
+++
{bdg-secondary}`trl`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth
:link: unsloth
:link-type: doc
GRPO training on instruction following and reasoning environments.
+++
{bdg-secondary}`unsloth` {bdg-secondary}`single-gpu` {bdg-secondary}`30 min`
:::


:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` NeMo Customizer
:link-type: doc
*Coming soon*
+++
{bdg-secondary}`nemo customizer` {bdg-warning}`in progress`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` VeRL
:link-type: doc
*Coming soon*
+++
{bdg-secondary}`verl` {bdg-warning}`in progress`
:::

::::

## SFT & DPO

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Offline Training with Rollouts
:link: offline-training-w-rollouts
:link-type: doc
Transform rollouts into training data for supervised fine-tuning (SFT) and direct preference optimization (DPO).
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::