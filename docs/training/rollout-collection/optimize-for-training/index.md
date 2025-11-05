(training-rollout-optimize)=

# Optimize for Training

Maximize rollout generation throughput when building large-scale training datasets.

:::{seealso}
New to rollout collection? Complete {doc}`../../../get-started/collecting-rollouts` first to get servers running.
:::

---

## Optimization Workflow

Follow this three-step process to systematically identify and resolve throughput bottlenecks.

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` 1. Identify Bottleneck
:link: identify-bottleneck
:link-type: doc

Run diagnostic tests to determine if parallelism, model server, or verification is limiting throughput.
+++
{bdg-primary}`Start here`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` 2. Tune Parallelism
:link: tune-parallelism
:link-type: doc

Find optimal `num_samples_in_parallel` value using NeMo Gym's semaphore-based concurrency control.
+++
{bdg-secondary}`Most impactful`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` 3. Training Throughput at Scale
:link: production-scale
:link-type: doc

Use NeMo Gym's built-in features for large-scale generation: resume, verification optimization, throughput tracking, and data quality validation.
+++
{bdg-secondary}`Advanced`
:::

::::

---

## Quick Reference

Use this table to quickly navigate to the optimization guide that matches your specific goal.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Goal
  - Guide
* - **Diagnose slow collection**
  - {doc}`identify-bottleneck` → Run throughput tests
* - **Configure concurrency**
  - {doc}`tune-parallelism` → Find optimal value
* - **Optimize verification**
  - {doc}`production-scale` → Caching and fast mode
* - **Resume interrupted runs**
  - {doc}`production-scale` → Append mode behavior
* - **Track throughput**
  - {doc}`production-scale` → Samples/sec and tokens/sec
* - **Quick quality checks**
  - {doc}`production-scale` → Smoke tests for broken verification
```

---

## Next Steps

Start with **{doc}`identify-bottleneck`** to diagnose what's limiting your throughput, then proceed to **{doc}`tune-parallelism`** for systematic optimization.

After maximizing throughput, tune data characteristics with **{doc}`../sampling-strategies/index`** (temperature, diversity, num_repeats).

:::{button-ref} identify-bottleneck
:color: primary
:outline:
:ref-type: doc

Start with Identify Bottleneck →
:::

---

```{toctree}
:maxdepth: 1
:hidden:

identify-bottleneck
tune-parallelism
production-scale
```
