(tutorials-index)=

# NeMo Gym Tutorials

Hands-on learning experiences that guide you through building, training, and deploying AI agents with NeMo Gym.

:::{tip}
**New to NeMo Gym?** Start with the {doc}`Get Started <../get-started/index>` section for a guided tutorial experience from installation through your first verified agent. Return here after completing those tutorials to learn about advanced topics like rollout collection and training data generation.
:::

:::{tip}
**Need a quick definition?** Check the {doc}`Glossary <../resources/glossary>` for essential terminology.
:::

---

## Rollout Collection and Training Data

Master rollout generation and training data preparation for RL, SFT, and DPO.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Offline Training with Rollouts
:link: offline-training-w-rollouts
:link-type: doc
Transform rollouts into training data for supervised fine-tuning (SFT) and direct preference optimization (DPO).
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

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Separate Policy and Judge Models
:link: separate-policy-and-judge-models
:link-type: doc
Configure multiple model servers for different roles—policy for generation, judge for verification. A production pattern for AI training and evaluation.
+++
{bdg-secondary}`configuration` {bdg-secondary}`multi-model` {bdg-secondary}`production-patterns`
:::

::::
