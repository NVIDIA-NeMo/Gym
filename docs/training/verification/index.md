(training-verification)=

# Verification for Training

Design effective reward signals and verification strategies that drive successful reinforcement learning. Learn reward shaping patterns, verification approaches, and multi-objective scoring for training.

You understand basic verification from {doc}`Get Started <../../get-started/verifying-agent-results>`—now learn advanced reward design for production training workflows.


## When You Need This

Use this section when you need to:

* **Design reward signals** - Create effective rewards that guide training toward desired behaviors
* **Shape rewards for RL** - Move beyond binary success/failure to nuanced scoring
* **Combine multiple signals** - Balance multiple objectives or metrics in a single reward
* **Choose verification patterns** - Understand tradeoffs between different verification approaches

:::{seealso}
For deep understanding of verification theory, see {doc}`../../about/concepts/verifying-agent-results`.
:::

:::{note}
**Building custom verifiers from scratch?** That's covered in the Environments section *(coming soon)*. This section focuses on reward design using existing verifiers.
:::


## Guides and References

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item-card} {octicon}`trophy;1.5em;sd-mr-1` Reward Shaping
:link: reward-shaping
:link-type: doc

**How-to guide** for designing effective reward signals that drive training. Learn binary, continuous, sparse, and dense reward patterns.
+++
{bdg-secondary}`how-to` {bdg-secondary}`rewards` {bdg-secondary}`rl`
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Multi-Objective Scoring
:link: multi-objective-scoring
:link-type: doc

**How-to guide** for combining multiple reward signals or balancing conflicting objectives in training.
+++
{bdg-secondary}`how-to` {bdg-secondary}`multi-objective` {bdg-secondary}`composite`
:::

:::{grid-item-card} {octicon}`bookmark;1.5em;sd-mr-1` Verification Patterns
:link: verification-patterns
:link-type: doc

**Reference** catalog of verification approaches with examples, tradeoffs, and when to use each pattern.
+++
{bdg-secondary}`reference` {bdg-secondary}`patterns` {bdg-secondary}`catalog`
:::

::::


## Quick Selection Guide

Choose verification pattern based on your training objective:

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Training Goal
  - Verification Pattern
  - Reward Type
* - **SFT (Correct Behavior)**
  - Binary or high-threshold continuous
  - 1.0 for correct, 0.0 for incorrect
* - **DPO (Preference Learning)**
  - Continuous with clear differences
  - Score pairs with quality gap ≥ 0.1
* - **RL (Exploration)**
  - Shaped continuous rewards
  - Partial credit for progress
* - **Multi-Task Training**
  - Multi-objective scoring
  - Weighted combination of signals
```

See {doc}`reward-shaping` for detailed guidance.


## Reward Design Principles

Key principles for effective reward signals:

```{list-table}
:header-rows: 1
:widths: 25 75

* - Principle
  - Why It Matters
* - **Aligned with goal**
  - Rewards must match what you want the model to learn. Misaligned rewards lead to unintended behaviors.
* - **Discriminative**
  - Good and bad behaviors need different scores. Rewards that are always 0.5 provide no learning signal.
* - **Consistent**
  - Similar behaviors should get similar rewards. Inconsistent scoring confuses training.
* - **Scalable**
  - Verification must be fast enough for large-scale collection (< 100ms preferred).
```

See {doc}`reward-shaping` for implementation patterns.


## Verification in the Training Pipeline

```
Rollout Generation
    ↓
[Agent Execution]        ← Tools, reasoning, response
    ↓
[Verification]           ← Score performance
    ↓
    Reward Signal        → to training framework
```

**Flow**: Generated rollout → Verification function → Reward score → Training data

**Next**: {doc}`../data-quality/index` for filtering by reward thresholds


## Common Verification Patterns

### Binary Verification
```python
# Simple correct/incorrect
reward = 1.0 if answer == expected else 0.0
```
**Use for**: SFT data, exact match tasks, pass/fail scenarios

### Continuous Scoring
```python
# Partial credit for quality
reward = calculate_similarity(answer, expected)  # 0.0-1.0
```
**Use for**: DPO pairs, quality-based training, nuanced evaluation

### Multi-Objective
```python
# Balance multiple goals
reward = 0.6 * correctness + 0.3 * efficiency + 0.1 * style
```
**Use for**: Complex tasks, multiple criteria, weighted objectives

See {doc}`verification-patterns` for complete catalog.


## Related Topics

### Data Pipeline

* {doc}`../rollout-collection/index` - Generate rollouts with verification
* {doc}`../data-quality/index` - Filter based on reward thresholds
* {doc}`../datasets/index` - Include rewards in training formats

### Custom Verification

* **Building Custom Resource Servers** *(coming soon)* - Implement custom verifiers from scratch
* {doc}`../../about/concepts/verifying-agent-results` - Deep dive on verification theory


## Next Steps

:::{button-ref} reward-shaping
:color: primary
:outline:
:ref-type: doc

Start with Reward Shaping →
:::

:::{tip}
**Not sure which verification pattern to use?** Check {doc}`verification-patterns` reference for a decision guide based on your domain and training objective.
:::

```{toctree}
:hidden:
:maxdepth: 1

reward-shaping
verification-patterns
multi-objective-scoring
```

