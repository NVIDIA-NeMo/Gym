(training-data-quality)=

# Data Quality for Training

Ensure high-quality training data through filtering, curation, and balancing strategies. Learn to assess quality metrics, remove poor rollouts, and create balanced datasets for effective training.

Generated rollouts vary in quality—now learn to systematically curate training datasets that drive model improvement.


## When You Need This

Use this section when you need to:

* **Filter low-quality rollouts** - Remove poor examples before training
* **Monitor data quality** - Track quality metrics during generation
* **Balance datasets** - Ensure diverse task representation and difficulty distribution
* **Set quality thresholds** - Determine appropriate reward cutoffs for your use case

:::{note}
**Quality over quantity**: Training on fewer high-quality examples beats training on many noisy ones. These guides help you curate effectively.
:::


## Guides and References

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filtering Strategies
:link: filtering-strategies
:link-type: doc

**How-to guide** for filtering rollouts based on quality thresholds, success criteria, and domain-specific requirements.
+++
{bdg-secondary}`how-to` {bdg-secondary}`filtering` {bdg-secondary}`thresholds`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Quality Metrics
:link: quality-metrics
:link-type: doc

**How-to guide** for tracking and monitoring quality during rollout collection using reward distributions and statistics.
+++
{bdg-secondary}`how-to` {bdg-secondary}`metrics` {bdg-secondary}`monitoring`
:::

:::{grid-item-card} {octicon}`rows;1.5em;sd-mr-1` Dataset Balancing
:link: dataset-balancing
:link-type: doc

**How-to guide** for balancing task types, difficulty levels, and diversity to prevent overfitting and improve generalization.
+++
{bdg-secondary}`how-to` {bdg-secondary}`balancing` {bdg-secondary}`diversity`
:::

::::


## Quality Pipeline

Data quality workflow in the training pipeline:

```
Generated Rollouts (raw)
    ↓
[1. Analyze Distributions]  ← reward, length, task type
    ↓
[2. Apply Filters]          ← min reward, success rate, validity
    ↓
[3. Balance Dataset]        ← task diversity, difficulty distribution
    ↓
Curated Training Data       → to datasets/prepare-for-training
```

**Previous**: {doc}`../rollout-collection/index` for generation  
**Next**: {doc}`../datasets/index` for formatting


## Quick Selection Guide

Choose quality strategy based on your training approach:

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Training Type
  - Quality Strategy
  - Typical Thresholds
* - **SFT**
  - High-quality filtering only
  - reward ≥ 0.8, strict success criteria
* - **DPO**
  - Filter pairs with sufficient quality gap
  - quality_difference ≥ 0.1-0.2
* - **RL**
  - Keep diverse quality range
  - reward ≥ 0.3, focus on distribution
* - **Curriculum Learning**
  - Balance by difficulty
  - Stratify easy/medium/hard tasks
```

See {doc}`filtering-strategies` for implementation details.


## Quality Metrics to Track

Key metrics for training data quality:

```{list-table}
:header-rows: 1
:widths: 25 50 25

* - Metric
  - What It Measures
  - Good Target
* - **Average Reward**
  - Overall quality of rollouts
  - ≥ 0.7 for SFT, varied for RL
* - **Success Rate**
  - Percentage passing threshold
  - ≥ 80% for SFT
* - **Reward Distribution**
  - Diversity of quality levels
  - Not all 0.0 or 1.0
* - **Length Distribution**
  - Consistency of interactions
  - No extreme outliers
* - **Task Diversity**
  - Coverage of task types
  - Balanced representation
```

See {doc}`quality-metrics` for tracking guidance.


## Filtering Patterns

### Conservative Filtering (SFT)
```python
# Keep only high-quality examples
filtered = [r for r in rollouts if r['reward'] >= 0.8 and r['success']]
```
**Result**: Smaller, cleaner dataset. Good for supervised fine-tuning.

### Permissive Filtering (RL)
```python
# Keep diverse quality range
filtered = [r for r in rollouts if r['reward'] >= 0.3]
```
**Result**: Larger, diverse dataset. Good for RL exploration.

### Pair Filtering (DPO)
```python
# Keep pairs with quality difference
pairs = [(r1, r2) for r1, r2 in pairs if abs(r1['reward'] - r2['reward']) >= 0.1]
```
**Result**: Preference pairs with clear winners. Good for DPO training.

See {doc}`filtering-strategies` for complete examples.


## Related Topics

### Data Pipeline

* {doc}`../rollout-collection/index` - Generate rollouts (before quality filtering)
* {doc}`../verification/index` - Design rewards that enable quality filtering
* {doc}`../datasets/index` - Format filtered data for training

### Analysis Tools

* `ng_viewer` - Interactive rollout viewer for quality inspection (documented in Get Started)
* `ng_prepare_data` - Dataset validation and statistics (see {doc}`../datasets/validate-format`)


## Next Steps

:::{button-ref} filtering-strategies
:color: primary
:outline:
:ref-type: doc

Start with Filtering Strategies →
:::

:::{tip}
**Unsure about quality thresholds?** Start with {doc}`quality-metrics` to understand your data distribution, then apply filters based on observed quality patterns.
:::

```{toctree}
:hidden:
:maxdepth: 1

filtering-strategies
quality-metrics
dataset-balancing
```

