(training-overview)=

# Training

Scale up your training data generation workflows and integrate with RL frameworks. This section focuses on practical guidance for generating high-quality training data at production scale.

You have completed {doc}`Get Started <../get-started/index>` and understand rollout collection basics—now learn to optimize, curate, and prepare data for reinforcement learning frameworks.

## When You Need This Section

This section is for practitioners who need to:

* **Scale up data generation** - Generate thousands or millions of rollouts for training
* **Ensure data quality** - Filter, curate, and balance training datasets
* **Shape rewards effectively** - Design verification strategies that produce strong training signals
* **Integrate with RL frameworks** - Connect rollouts to NeMo-RL, VeRL, OpenRLHF, TRL, or custom frameworks
* **Optimize throughput** - Maximize rollouts per hour for production workflows

:::{tip}
**Looking for system configuration or performance tuning?** Those topics are covered in the Setup and Deployment section (coming soon).
:::

## Training Data Pipeline

Follow the training data pipeline from generation to framework integration:

::::{grid} 1 1 2 2
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
  - {doc}`Rollout Collection / Optimize for Training <rollout-collection/optimize-for-training>`
* - Design better reward signals
  - {doc}`Verification / Reward Shaping <verification/reward-shaping>`
* - Improve training data quality
  - {doc}`Data Quality / Filtering Strategies <data-quality/filtering-strategies>`
* - Prepare data for SFT or DPO
  - {doc}`Datasets / Prepare for Training <datasets/prepare-for-training>`
* - Connect to your RL framework
  - {doc}`Integration / Framework Comparison <integration/framework-comparison>` then select your framework
* - Understand verification approaches
  - {doc}`Verification / Verification Patterns <verification/verification-patterns>` (reference)
```

## Training Workflow Patterns

### Pattern 1: SFT Data Generation

Generate high-quality demonstration data for supervised fine-tuning:

```yaml
# Optimized for consistency
num_samples_in_parallel: 20
responses_create_params:
  temperature: 0.2  # Low for consistent behavior
  
# Then filter for quality
min_reward_threshold: 0.8
```

**Guides**: {doc}`rollout-collection/sampling-strategies` → {doc}`data-quality/filtering-strategies` → {doc}`datasets/prepare-for-training`

### Pattern 2: DPO Pair Generation

Generate diverse pairs for preference optimization:

```yaml
# Optimized for diversity
num_repeats: 2  # Multiple samples per task
responses_create_params:
  temperature: 0.7  # Higher for variation
  
# Then create preference pairs
min_quality_difference: 0.1
```

**Guides**: {doc}`rollout-collection/sampling-strategies` → {doc}`data-quality/filtering-strategies` → {doc}`datasets/prepare-for-training`

### Pattern 3: RL Training Data

Generate rollouts with shaped rewards for reinforcement learning:

```yaml
# Balanced approach
num_samples_in_parallel: 10
responses_create_params:
  temperature: 0.5  # Moderate exploration
  
# Use shaped rewards
reward_shaping: continuous  # 0.0-1.0 range
```

**Guides**: {doc}`verification/reward-shaping` → {doc}`rollout-collection/optimize-for-training` → {doc}`integration/nemo-rl` (or your framework)

## Related Documentation

### Concepts and Background

For understanding **why** these practices matter:

* {doc}`../about/concepts/rollout-collection-fundamentals` - Deep dive on rollout generation
* {doc}`../about/concepts/verifying-agent-results` - Theory behind verification and rewards
* {doc}`../about/concepts/configuration-system` - Configuration hierarchy explained

### Tutorials and Learning

For **learning-oriented** experiences:

* {doc}`../get-started/collecting-rollouts` - Your first rollout collection (prerequisite)
* {doc}`../tutorials/offline-training-w-rollouts` - End-to-end SFT/DPO tutorial

### System Configuration

For **system-level** setup (not training-specific):

* Configuration Management *(coming soon)* - Multi-model setups, environment configs, parameter reference
* Performance & Scaling *(coming soon)* - System-level throughput optimization, distributed generation

## Next Steps

:::{button-ref} rollout-collection/index
:color: primary
:outline:
:ref-type: doc

Start with Rollout Collection →
:::

:::{tip}
**New to training workflows?** We recommend starting with Rollout Collection to understand data generation optimization, then moving to Data Quality to ensure your datasets are production-ready.
:::
