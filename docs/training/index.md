(training-overview)=

# About Training

Generate training data by orchestrating agent interactions with tools and environments. Use these guides to integrate Gym into your RL training pipelines and prepare high-quality rollout data.

---

## Training Data Pipeline

NeMo Gym is a **data generation** tool. Unlike typical ML workflows where you start with datasets, Gym **produces** training datasets as output.

```{mermaid}
flowchart LR
    A["Task Prompts"] --> B["Rollout Collection"]
    B --> C["Verification"]
    C --> D["Training Dataset"]
    
    E["Training Framework"] -.->|"integrates with"| B
    
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
    classDef external fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    
    class A input
    class B,C process
    class D output
    class E external
```

Your inputs are task definitions (resource servers) and prompts. The pipeline produces verified rollouts formatted for your training objective (RL, SFT, or DPO).

---

## Integrate

Connect NeMo Gym to your training framework to collect rollouts with token IDs, log probabilities, and rewards.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL
:link: /tutorials/integrate-training-frameworks/train-with-nemo-rl
:link-type: doc
Native integration with NeMo RL for single-node and multi-node training runs.
+++
{bdg-info}`tutorial`
{bdg-primary}`recommended`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Custom Frameworks
:link: integrate/index
:link-type: doc
Integrate Gym into custom training pipelines using OpenAI-compatible endpoints.
+++
{bdg-success}`how-to`
{bdg-warning}`advanced`
:::

::::

---

## Collect Rollouts

Generate and process rollouts for different training objectives.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Get Started with Rollouts
:link: /get-started/rollout-collection
:link-type: doc
Generate your first rollouts using resource servers and view results with the rollout viewer.
+++
{bdg-info}`tutorial`
{bdg-secondary}`ng_collect_rollouts`
:::

:::{grid-item-card} {octicon}`sliders;1.5em;sd-mr-1` Configure Sampling
:link: rollout-collection/configure-sampling
:link-type: doc
Set temperature, repeats, and filtering for RL, SFT, and DPO objectives.
+++
{bdg-success}`how-to`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Optimize Performance
:link: rollout-collection/index
:link-type: doc
Profile collection, tune parallelism, and scale for production.
+++
{bdg-success}`how-to`
:::

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Offline Training Data
:link: /tutorials/offline-training-w-rollouts
:link-type: doc
Transform rollouts into training data for SFT and DPO.
+++
{bdg-info}`tutorial`
{bdg-secondary}`sft`
{bdg-secondary}`dpo`
:::

::::

---

## Verify and Format Datasets

Design reward signals and prepare training data.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Verification Patterns
:link: verification/index
:link-type: doc
Implement multi-verifier training, custom scoring, and reward logic.
+++
{bdg-success}`how-to`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Prepare Training Data
:link: datasets/prepare-data
:link-type: doc
Use `ng_prepare_data` to validate, format, and add agent routing.
+++
{bdg-success}`how-to`
:::

:::{grid-item-card} {octicon}`cloud-upload;1.5em;sd-mr-1` HuggingFace Integration
:link: datasets/huggingface-integration
:link-type: doc
Upload and download datasets from HuggingFace Hub.
+++
{bdg-success}`how-to`
:::

::::

---

## Learn More

Explore the architecture and ecosystem behind Gym's training integrations.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Integration Architecture
:link: /about/concepts/training-integration-architecture
:link-type: doc
How Gym connects to training frameworks, request lifecycle, and token alignment.
+++
{bdg-secondary}`explanation`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Framework Ecosystem
:link: /about/ecosystem
:link-type: doc
Existing training framework integrations and compatibility information.
+++
{bdg-secondary}`reference`
:::

::::
