(training-overview)=

# Training

Generate high-quality training data at scale with optimized rollout collection, verification, and formatting.

## Training Data Pipeline

Follow the training data pipeline from resource server selection to framework integration:

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Resource Servers
:link: resource-servers/index
:link-type: doc

Choose a resource server that provides tools, datasets, and verification for your training task.
+++
{bdg-secondary}`server-selection` {bdg-secondary}`tasks` {bdg-secondary}`domains`
:::

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

Validate that verification works correctly and customize reward signals for your training needs.
+++
{bdg-secondary}`validation` {bdg-secondary}`rewards` {bdg-secondary}`custom-patterns`
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

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Training Integration
:link: integration/index
:link-type: doc

Pass your rollouts to RL training frameworks. NeMo-RL integration with GRPO, SFT, and DPO guides.
+++
{bdg-secondary}`nemo-rl` {bdg-secondary}`frameworks` {bdg-secondary}`integration`
:::

::::

## Quick Decision Guide

Not sure where to start? Choose based on your current need:

```{list-table}
:header-rows: 1
:widths: 40 60

* - If You Need To...
  - Start Here
* - Choose a resource server
  - {doc}`Resource Servers <resource-servers/index>` (by task type, training algorithm)
* - Generate training data faster
  - {doc}`Rollout Collection / Optimize for Training <rollout-collection/optimize-for-training/index>`
* - Validate verification works
  - {doc}`Verification / Validate <verification/validate-verification>` (check reward signals)
* - Improve training data quality
  - {doc}`Data Quality <data-quality/index>`
* - Prepare data for SFT or DPO
  - {doc}`Datasets / Prepare for Training <datasets/prepare-for-training>`
* - Pass data to your training framework
  - {doc}`Training Integration <integration/index>` (NeMo-RL, VeRL, OpenRLHF, TRL)
* - Build custom verification
  - {doc}`Verification / Custom Patterns <verification/custom-patterns-cookbook>` (advanced)
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

**Guides**: {doc}`rollout-collection/optimize-for-training/index` → {doc}`data-quality/index` → {doc}`datasets/prepare-for-training`

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

**Guides**: {doc}`rollout-collection/optimize-for-training/index` → {doc}`data-quality/index` → {doc}`datasets/prepare-for-training`

:::

:::{tab-item} PPO Training Data

Generate rollouts with continuous rewards for reinforcement learning:

```yaml
# Balanced approach
num_samples_in_parallel: 10
responses_create_params:
  temperature: 0.5  # Moderate exploration
```

**Guides**: {doc}`rollout-collection/optimize-for-training/index` → {doc}`data-quality/index` → {doc}`integration/index`

:::

::::

## Next Steps

We recommend starting with **Resource Servers** to choose the right task domain and verification, then moving to **Rollout Collection** to generate training data at scale.

:::{button-ref} resource-servers/index
:color: primary
:outline:
:ref-type: doc

Choose a Resource Server →
:::
