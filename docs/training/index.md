(training-overview)=

# Training

Scale up your training data generation workflows and integrate with RL frameworks. This section focuses on practical guidance for generating high-quality training data at production scale.

## Training Data Pipeline

Follow the training data pipeline from generation to framework integration:

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection
:link: rollout-collection/index
:link-type: doc

Generate training rollouts at scale with optimized sampling strategies and parallelization.
+++
{bdg-secondary}`data-generation` {bdg-secondary}`parallelism` {bdg-secondary}`throughput`
:::

:::{grid-item-card} {octicon}`trophy;1.5em;sd-mr-1` Verification
:link: verification/index
:link-type: doc

Design reward signals and verification strategies that drive effective training.
+++
{bdg-secondary}`rewards` {bdg-secondary}`scoring` {bdg-secondary}`reward-shaping`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Data Quality
:link: data-quality/index
:link-type: doc

Filter, curate, and balance rollouts to ensure high-quality training datasets.
+++
{bdg-secondary}`filtering` {bdg-secondary}`curation` {bdg-secondary}`quality-metrics`
:::

:::{grid-item-card} {octicon}`package-dependencies;1.5em;sd-mr-1` Datasets
:link: datasets/index
:link-type: doc

Organize, validate, and prepare datasets in formats for RL training frameworks.
+++
{bdg-secondary}`formats` {bdg-secondary}`validation` {bdg-secondary}`sft-dpo`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integration
:link: integration/index
:link-type: doc

Connect your training data to NeMo-RL, VeRL, OpenRLHF, TRL, and custom frameworks.
+++
{bdg-secondary}`rl-frameworks` {bdg-secondary}`nemo-rl` {bdg-secondary}`verl`
:::

::::

## Quick Decision Guide

Not sure where to start? Choose based on your current need:

```{list-table}
:header-rows: 1
:widths: 40 60

* - If You Need To...
  - Start Here
* - Generate training data faster
  - {doc}`Rollout Collection / Optimize for Training <rollout-collection/optimize-for-training/index>`
* - Design better reward signals
  - {doc}`Verification / Reward Shaping <verification/reward-shaping>`
* - Improve training data quality
  - {doc}`Data Quality <data-quality/index>`
* - Prepare data for SFT or DPO
  - {doc}`Datasets / Prepare for Training <datasets/prepare-for-training>`
* - Connect to your RL framework
  - {doc}`Integration / Framework Comparison <integration/framework-comparison>` then select your framework
* - Build custom verification
  - {doc}`Verification / Custom Patterns <verification/custom-patterns-cookbook>` (cookbook)
```

## Training Workflow Patterns

Common end-to-end workflows combining data generation, quality filtering, and framework integration for different training objectives.

::::{tab-set}

:::{tab-item} SFT Data Generation

Generate high-quality demonstration data for supervised fine-tuning:

```yaml
# Optimized for consistency
num_samples_in_parallel: 20
responses_create_params:
  temperature: 0.2  # Low for consistent behavior
  
# Then filter for quality
min_reward_threshold: 0.8
```

**Guides**: {doc}`rollout-collection/sampling-strategies/sft` → {doc}`data-quality/index` → {doc}`datasets/prepare-for-training`

:::

:::{tab-item} DPO Pair Generation

Generate diverse pairs for preference optimization:

```yaml
# Optimized for diversity
num_repeats: 2  # Multiple samples per task
responses_create_params:
  temperature: 0.7  # Higher for variation
  
# Then create preference pairs
min_quality_difference: 0.1
```

**Guides**: {doc}`rollout-collection/sampling-strategies/dpo` → {doc}`data-quality/index` → {doc}`datasets/prepare-for-training`

:::

:::{tab-item} RL Training Data

Generate rollouts with shaped rewards for reinforcement learning:

```yaml
# Balanced approach
num_samples_in_parallel: 10
responses_create_params:
  temperature: 0.5  # Moderate exploration
  
# Use shaped rewards
reward_shaping: continuous  # 0.0-1.0 range
```

**Guides**: {doc}`verification/reward-shaping` → {doc}`rollout-collection/optimize-for-training/index` → {doc}`integration/nemo-rl` (or your framework)

:::

::::

## Next Steps

We recommend starting with **Rollout Collection** to understand data generation optimization, then moving to **Data Quality** to ensure your datasets are production-ready.

:::{button-ref} rollout-collection/index
:color: primary
:outline:
:ref-type: doc

Start with Rollout Collection →
:::