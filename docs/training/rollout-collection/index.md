(training-rollout-collection)=

# Rollout Collection for Training

Generate training rollouts at scale with optimized sampling strategies, parallelization, and collection patterns tailored to your training objectives.

:::{seealso}
For deep understanding of rollout fundamentals, see {ref}`concepts-rc-fundamentals`.
:::

## Topics

Explore the three core aspects of training data collection: performance optimization, sampling configuration, and collection patterns.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Optimize for Training
:link: optimize-for-training/index
:link-type: doc

Maximize training data generation throughput with configuration tuning and parallelization strategies.
+++
{bdg-secondary}`performance` {bdg-secondary}`parallelism` {bdg-secondary}`throughput`
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Sampling Strategies
:link: sampling-strategies/index
:link-type: doc

Configure temperature, diversity, and sampling parameters for SFT, DPO, RL, evaluation, and research.
+++
{bdg-secondary}`temperature` {bdg-secondary}`diversity` {bdg-secondary}`sampling`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Collection Patterns
:link: collection-patterns/index
:link-type: doc

Common collection patterns and configurations for different training scenarios and use cases.
+++
{bdg-secondary}`patterns` {bdg-secondary}`examples` {bdg-secondary}`commands`
:::

::::

## Quick Selection Guide

Choose your starting point based on your goal:

```{list-table}
:header-rows: 1
:widths: 50 50

* - Training Goal
  - Recommended Configuration
* - **SFT (Supervised Fine-Tuning)**
  - Low temperature (0.1-0.3), high parallelism, single sample per task. See {doc}`sampling-strategies/index`
* - **DPO (Direct Preference Optimization)**
  - Higher temperature (0.6-0.8), multiple samples per task (2+), focus on diversity. See {doc}`sampling-strategies/index`
* - **RL (Reinforcement Learning)**
  - Moderate temperature (0.4-0.6), shaped rewards, balanced exploration. See {doc}`collection-patterns/index`
* - **Maximize Speed**
  - Tune parallelism, optimize model server, reduce verification overhead. See {doc}`optimize-for-training/index`
```

## Collection Process Overview

The training data collection process:

```
Input Dataset (JSONL)
    ↓
[1. Configure Sampling]  ← temperature, top-p, diversity
    ↓
[2. Set Parallelism]     ← num_samples_in_parallel
    ↓
[3. Generate Rollouts]   ← ng_collect_rollouts
    ↓
[4. Verify & Score]      ← resource server verification
    ↓
Output Rollouts (JSONL)  → to Data Quality filtering
```

```{toctree}
:hidden:
:maxdepth: 2

optimize-for-training/index
sampling-strategies/index
collection-patterns/index
```
