(training-rollout-sampling)=

# Sampling Strategies

Configure temperature, diversity, and sampling parameters to match your training objective.

Different training algorithms need different data characteristics—learn how to tune rollout generation for SFT, DPO, RL, and evaluation.

**What makes NeMo Gym sampling strategies effective**: Automatic metric aggregation from verification, guaranteed consecutive grouping for DPO pairing (`num_repeats`), parameter override flexibility for per-task customization, and built-in retry handling for high-parallelism collection.

---

## Quick Strategy Guide

```{list-table}
:header-rows: 1
:widths: 20 15 15 15 35

* - Training Type
  - Temperature
  - Num Repeats
  - Parallelism
  - Goal
* - **SFT**
  - Low
  - 1
  - High
  - Consistent demonstrations at scale
* - **DPO**
  - Higher
  - 3-4
  - Medium
  - Diverse preference pairs for comparison
* - **RL**
  - Moderate
  - 1
  - Medium-High
  - Balance exploration and exploitation
```

---

## Guides and References

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Understanding Parameters
:link: parameters
:link-type: doc

**Explanation** of temperature, top_p, and num_repeats—what they control and how they affect rollout characteristics.
+++
{bdg-secondary}`explanation` {bdg-secondary}`fundamentals`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` SFT Sampling Strategy
:link: sft
:link-type: doc

**How-to guide** for supervised fine-tuning: consistent, high-quality demonstrations at scale.
+++
{bdg-secondary}`how-to` {bdg-secondary}`sft` {bdg-secondary}`consistency`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` DPO Sampling Strategy
:link: dpo
:link-type: doc

**How-to guide** for Direct Preference Optimization: generating diverse preference pairs with quality differences.
+++
{bdg-secondary}`how-to` {bdg-secondary}`dpo` {bdg-secondary}`diversity`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` RL Sampling Strategy
:link: rl
:link-type: doc

**How-to guide** for reinforcement learning: balancing exploration and exploitation with iterative collection.
+++
{bdg-secondary}`how-to` {bdg-secondary}`rl` {bdg-secondary}`exploration`
:::


::::

---

## Strategy Selection Flow

Use this flowchart to determine which sampling strategy matches your training goal.

~~~{mermaid}
flowchart TD
    Start[What is your goal?] --> Training{Training or<br/>Evaluation?}
    
    Training -->|Training| Algorithm{Which algorithm?}
    
    Algorithm -->|SFT| SFT[Use SFT Strategy<br/>low temp, single sample]
    Algorithm -->|DPO| DPO[Use DPO Strategy<br/>higher temp, multiple repeats]
    Algorithm -->|RL| RL[Use RL Strategy<br/>moderate temp, iterative]
    
    SFT --> Filter[Filter for high rewards]
    DPO --> Pair[Create preference pairs]
    RL --> Iterate[Iterative collection<br/>+ policy updates]
~~~

---

## Quick Command Templates

Copy and customize these templates for each training type. Configure `temperature`, `num_repeats`, and `num_samples_in_parallel` based on your specific needs.

::::{tab-set}

:::{tab-item} SFT

```bash
ng_collect_rollouts \
    +agent_name=<agent_name> \
    +input_jsonl_fpath=<input_file> \
    +output_jsonl_fpath=<output_file> \
    +responses_create_params.temperature=0.2 \
    +num_samples_in_parallel=20
```

**Typical settings**: Low temperature (0.1-0.3) for consistency, high parallelism (15-30) for scale. Built-in retry logic handles rate limits automatically.

:::

:::{tab-item} DPO

```bash
ng_collect_rollouts \
    +agent_name=<agent_name> \
    +input_jsonl_fpath=<input_file> \
    +output_jsonl_fpath=<output_file> \
    +responses_create_params.temperature=0.7 \
    +num_repeats=3 \
    +num_samples_in_parallel=10
```

**Typical settings**: Higher temperature (0.6-0.8) for diversity, multiple repeats (3-4) creating consecutive groups for pairing, moderate parallelism

:::

:::{tab-item} RL

```bash
ng_collect_rollouts \
    +agent_name=<agent_name> \
    +input_jsonl_fpath=<input_file> \
    +output_jsonl_fpath=<output_file> \
    +responses_create_params.temperature=0.5 \
    +num_samples_in_parallel=15
```

**Typical settings**: Moderate temperature (0.4-0.6) for exploration/exploitation balance, medium-high parallelism

:::


::::

## Next Steps

Start with **{doc}`parameters`** to understand the fundamentals, then choose the strategy guide that matches your training objective.

:::{button-ref} parameters
:color: primary
:outline:
:ref-type: doc

Start with Understanding Parameters →
:::

```{toctree}
:hidden:
:maxdepth: 1

Sampling Parameters <parameters>
SFT <sft>
DPO <dpo>
RL <rl>
```
