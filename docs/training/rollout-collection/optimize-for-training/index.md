(training-rollout-optimize)=

# Optimize for Training

Maximize rollout generation throughput when building large-scale training datasets.

:::{seealso}
New to rollout collection? Complete {ref}`gs-collecting-rollouts` first to get servers running.
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
  - {ref}`training-rollout-optimize-bottleneck` → Run throughput tests
* - **Configure concurrency**
  - {ref}`training-rollout-optimize-parallelism` → Find optimal value
* - **Optimize verification**
  - {ref}`training-rollout-optimize-production` → Caching and fast mode
* - **Resume interrupted runs**
  - {ref}`training-rollout-optimize-production` → Append mode behavior
* - **Track throughput**
  - {ref}`training-rollout-optimize-production` → Samples/sec and tokens/sec
* - **Quick quality checks**
  - {ref}`training-rollout-optimize-production` → Smoke tests for broken verification
```

---

## Next Steps

Start with **{ref}`training-rollout-optimize-bottleneck`** to diagnose what's limiting your throughput, then proceed to **{ref}`training-rollout-optimize-parallelism`** for systematic optimization.

After maximizing throughput, tune data characteristics with **{ref}`training-rollout-sampling`** (temperature, diversity, num_repeats).

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
