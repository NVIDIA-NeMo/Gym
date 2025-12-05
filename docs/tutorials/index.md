(tutorials-index)=

# NeMo Gym Tutorials

Build, train, and deploy AI agents with NeMo Gym through hands-on guided experiences.

:::{tip}
**New to NeMo Gym?** Begin with the {doc}`Get Started <../get-started/index>` section for a guided tutorial from installation through your first verified agent. Return here afterward to learn about advanced topics like additional rollout collection methods and training data generation. You can find the project repository on [GitHub](https://github.com/NVIDIA-NeMo/Gym).
:::
---

## Integrate a Training Framework

Connect Gym to popular training frameworks for end-to-end model improvement.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`hubot;1.5em;sd-mr-1` Train with TRL (Offline)
:link: integrate-training-frameworks/train-with-trl
:link-type: doc
Use Hugging Face's TRL library for SFT, DPO, or GRPO training.
+++
{bdg-secondary}`offline` {bdg-secondary}`hugging-face`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: integrate-training-frameworks/train-with-nemo-rl
:link-type: doc
Use NVIDIA's NeMo RL for distributed on-policy training with NeMo 2.0 models.
+++
{bdg-secondary}`nvidia` {bdg-secondary}`multi-node`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Train with VeRL
:link: integrate-training-frameworks/train-with-verl
:link-type: doc
Use VeRL's Ray-based distributed training with flexible backend support.
+++
{bdg-secondary}`ray` {bdg-secondary}`multi-backend`
:::


::::

---

## Create Resource Servers

Build custom resource servers with tools, verification logic, and domain-specific functionality.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Weather API Server
:link: resource-servers/simple-tool-calling
:link-type: doc
Build a single-tool weather server with deterministic verification.
+++
{bdg-secondary}`single-tool`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Data Extraction Server
:link: resource-servers/multi-step-interactions
:link-type: doc
Build a multi-step server where agents query multiple data sources.
+++
{bdg-secondary}`multi-step`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Counter Game Server
:link: resource-servers/stateful-sessions
:link-type: doc
Build a stateful server where tools modify persistent state.
+++
{bdg-secondary}`stateful`
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` Math Verifier Server
:link: resource-servers/llm-as-judge
:link-type: doc
Build a math server with LLM-based answer verification.
+++
{bdg-secondary}`llm-judge`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Testing Server
:link: resource-servers/code-execution
:link-type: doc
Build a server that verifies code by executing test cases.
+++
{bdg-secondary}`code-exec`
:::

::::
