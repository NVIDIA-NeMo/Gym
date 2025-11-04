(training-rollout-collection)=

# Rollout Collection for Training

Generate training rollouts at scale with optimized sampling strategies, parallelization, and collection patterns tailored to your training objectives.

You have learned basic rollout collection in {doc}`Get Started <../../get-started/collecting-rollouts>`—now scale up to production training workflows with optimizations for SFT, DPO, and RL data generation.

---

## When You Need This

Use this section when you need to:

* **Generate large training datasets** - Thousands or millions of rollouts for RL, SFT, or DPO
* **Optimize generation speed** - Maximize throughput for your infrastructure
* **Control data characteristics** - Tune temperature, diversity, and sampling for your training approach
* **Choose collection strategies** - Understand tradeoffs between consistency, diversity, and efficiency

:::{seealso}
For deep understanding of rollout fundamentals, see {doc}`../../about/concepts/rollout-collection-fundamentals`.
:::

---

## Guides and References

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Optimize for Training
:link: optimize-for-training
:link-type: doc

**How-to guide** for maximizing training data generation throughput with configuration tuning and parallelization strategies.
+++
{bdg-secondary}`how-to` {bdg-secondary}`performance` {bdg-secondary}`parallelism`
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Sampling Strategies
:link: sampling-strategies
:link-type: doc

**How-to guide** for choosing temperature, diversity, and sampling parameters based on your training objective (SFT, DPO, RL).
+++
{bdg-secondary}`how-to` {bdg-secondary}`temperature` {bdg-secondary}`diversity`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Collection Patterns
:link: collection-patterns
:link-type: doc

**Reference** of common collection patterns and configurations for different training scenarios and use cases.
+++
{bdg-secondary}`reference` {bdg-secondary}`patterns` {bdg-secondary}`examples`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Reference
:link: configuration-reference
:link-type: doc

**Reference** of all `ng_collect_rollouts` parameters with descriptions, types, defaults, and examples.
+++
{bdg-secondary}`reference` {bdg-secondary}`parameters` {bdg-secondary}`config`
:::

::::

---

## Quick Selection Guide

Choose your starting point based on your goal:

```{list-table}
:header-rows: 1
:widths: 50 50

* - Training Goal
  - Recommended Configuration
* - **SFT (Supervised Fine-Tuning)**
  - Low temperature (0.1-0.3), high parallelism, single sample per task. See {doc}`sampling-strategies`
* - **DPO (Direct Preference Optimization)**
  - Higher temperature (0.6-0.8), multiple samples per task (2+), focus on diversity. See {doc}`sampling-strategies`
* - **RL (Reinforcement Learning)**
  - Moderate temperature (0.4-0.6), shaped rewards, balanced exploration. See {doc}`collection-patterns`
* - **Maximize Speed**
  - Tune parallelism, optimize model server, reduce verification overhead. See {doc}`optimize-for-training`
```

---

## Collection Pipeline Overview

The training data collection pipeline:

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

**Next**: {doc}`../data-quality/index` for filtering and curation

---

## Related Topics

### Configuration and Performance

* **Configuration Management** *(coming soon)* - System-level configuration patterns
* **Performance & Scaling** *(coming soon)* - System throughput optimization and distributed generation

### Data Pipeline

* {doc}`../verification/index` - Design reward signals for training
* {doc}`../data-quality/index` - Filter and curate generated rollouts
* {doc}`../datasets/index` - Prepare data for training frameworks

---

## Next Steps

:::{button-ref} optimize-for-training
:color: primary
:outline:
:ref-type: doc

Start with Optimize for Training →
:::

:::{tip}
**First time scaling up?** Start with {doc}`optimize-for-training` to understand parallelization and throughput, then move to {doc}`sampling-strategies` to tune data characteristics.
:::

