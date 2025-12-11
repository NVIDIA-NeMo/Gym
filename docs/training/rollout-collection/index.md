(training-rollout-collection-overview)=

# Rollout Collection

Generate rollouts at scale for RL, SFT, and DPO training objectives. Configure sampling strategies and optimize for production throughput.

---

## How It Works

Rollout collection is the core data generation step in NeMo Gym. Each rollout captures a complete agent interaction: the input prompt, model reasoning, tool calls, and a verification score.

```{mermaid}
flowchart TB
    subgraph Collection["Rollout Collection"]
        direction LR
        P["Prompts<br/>(JSONL)"] --> C["ng_collect_rollouts"]
        C --> R["Rollouts<br/>(with rewards)"]
    end
    
    subgraph Objectives["Training Objectives"]
        direction TB
        RL["RL<br/>(all rollouts)"]
        SFT["SFT<br/>(high-reward only)"]
        DPO["DPO<br/>(chosen/rejected pairs)"]
    end
    
    R --> RL
    R --> SFT
    R --> DPO
    
    classDef collection fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef objective fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    
    class P,C,R collection
    class RL,SFT,DPO objective
```

Different training objectives require different sampling strategies:

- **RL** — Generate diverse rollouts with exploration for policy optimization
- **SFT** — Filter to high-reward rollouts for supervised fine-tuning
- **DPO** — Create preference pairs from high/low reward rollouts

---

## Configure Sampling

Set temperature, repeats, and filtering for your training objective.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`sliders;1.5em;sd-mr-1` Configure Sampling Parameters
:link: configure-sampling
:link-type: doc

Temperature, top-p, and repeat settings for RL, SFT, and DPO objectives.
+++
{bdg-secondary}`ng_collect_rollouts`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filter and Transform Rollouts
:class-card: sd-card-placeholder

Post-process rollouts: filter by reward, create DPO pairs, format for training.
+++
{bdg-light}`planned`
:::

::::

---

## Process for Training

Prepare rollouts for your training framework.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Process Multi-Turn Rollouts
:link: process-multi-turn-rollouts
:link-type: doc

Handle token alignment across multi-turn interactions for correct gradient computation.
+++
{bdg-secondary}`tokens`
{bdg-secondary}`alignment`
:::

::::

---

## Optimize Performance

Scale rollout collection for production training runs.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Profile Collection
:link: profile-collection
:link-type: doc

Enable profiling to identify bottlenecks in verification and tool execution.
+++
{bdg-secondary}`profiling`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Tune Parallelism
:link: tune-parallelism
:link-type: doc

Configure `num_samples_in_parallel` and connection limits for optimal throughput.
+++
{bdg-secondary}`performance`
:::

::::

---

```{toctree}
:maxdepth: 1
:hidden:

configure-sampling
process-multi-turn-rollouts
profile-collection
tune-parallelism
```

