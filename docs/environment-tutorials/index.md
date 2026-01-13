(environment-tutorials-index)=
# Custom Environment Tutorials

Learn how to build custom training environments with NeMo Gym for various RL scenarios.

## Environment Patterns

NeMo Gym supports different environment patterns for different training objectives:

| Pattern | Description | Example Use Case |
|---------|-------------|------------------|
| **Single-step** | One model response per task | Math, Q&A |
| **Multi-step** | Sequential tool calls | Complex workflows |
| **Multi-turn** | Conversation with history | Dialogue agents |
| **User modeling** | Simulated user interactions | Customer service |

## Tutorials

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Creating a Training Environment
:link: creating-training-environment
:link-type: doc
Build a complete environment from scratch.
+++
{bdg-primary}`beginner` {bdg-secondary}`foundational`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Environments
:link: multi-step
:link-type: doc
Sequential tool calling workflows.
+++
{bdg-secondary}`multi-step` {bdg-secondary}`tools`
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Multi-Turn Environments
:link: multi-turn
:link-type: doc
Conversational training environments.
+++
{bdg-secondary}`multi-turn` {bdg-secondary}`conversation`
:::

:::{grid-item-card} {octicon}`people;1.5em;sd-mr-1` User Modeling
:link: user-modeling
:link-type: doc
Simulate users for dialogue training.
+++
{bdg-secondary}`user-modeling` {bdg-secondary}`dialogue`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Multi-Node Docker
:link: multi-node-docker
:link-type: doc
Distributed environment deployment.
+++
{bdg-secondary}`docker` {bdg-secondary}`distributed`
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-a-Judge
:link: llm-as-judge
:link-type: doc
LLM-based response verification.
+++
{bdg-secondary}`verification` {bdg-secondary}`llm-judge`
:::

:::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` RLHF with Reward Models
:link: rlhf-reward-models
:link-type: doc
Integrate learned reward models.
+++
{bdg-secondary}`rlhf` {bdg-secondary}`reward-models`
:::

::::

## Getting Started

New to NeMo Gym? Start with:
1. {doc}`/get-started/detailed-setup` — Install and run NeMo Gym
2. {doc}`/tutorials/creating-resource-server` — Build your first resources server
3. {doc}`creating-training-environment` — Create a complete training environment
