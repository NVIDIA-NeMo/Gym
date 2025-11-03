(tutorials-index)=

# NeMo Gym Tutorials

Hands-on learning experiences that guide you through building, training, and deploying AI agents with NeMo Gym.

:::{tip}
**Need a quick definition?** Check the {doc}`Glossary <../resources/glossary>` for essential terminology.
:::

---

## Getting Started

Learn core concepts and run your first agent in NeMo Gym.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Setup and Installation
:link: 02-setup
:link-type: doc
Get NeMo Gym installed and servers running with your first successful agent interaction.
+++
{bdg-primary}`beginner` {bdg-secondary}`installation`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Your First Agent
:link: 03-your-first-agent
:link-type: doc
Break down the agent workflow step-by-step and experiment with different inputs.
+++
{bdg-primary}`beginner` {bdg-secondary}`hands-on`
:::

::::

---

## Training

Understand verification and collect rollouts for RL, SFT, and DPO.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Verifying Agent Results
:link: 04-verifying-results
:link-type: doc
Understand how NeMo Gym evaluates agent performance and what verification means for training.
+++
{bdg-secondary}`verification` {bdg-secondary}`training`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection Fundamentals
:link: 05-rollout-collection
:link-type: doc
Master rollout generation - the foundation for understanding agent behavior and creating training data.
+++
{bdg-secondary}`data-generation` {bdg-secondary}`rollouts`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` SFT and DPO Training
:link: 07-sft-dpo-rollout-collection
:link-type: doc
Use generated rollouts to create training data for supervised fine-tuning and direct preference optimization.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::

---

<!-- ## Resource Servers

Build custom environments, tools, and verification systems.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Building Custom Resource Servers
:class-card: sd-border-0 sd-bg-light
*Coming soon* - Learn to create your own tools and verification systems, integrate with MCP and Docker, and perform dynamic prompting.
+++
{bdg-warning}`coming-soon` {bdg-secondary}`custom-environments`
:::

:::: -->

---

## Advanced Operations

Master configuration, testing, deployment, and scaling.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Management
:link: 09-configuration-guide
:link-type: doc
Master NeMo Gym's flexible configuration system to handle different environments, secrets, and deployment scenarios.
+++
{bdg-secondary}`configuration` {bdg-secondary}`deployment`
:::

::::
