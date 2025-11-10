(training-rollout-optimize-bottleneck)=

# Identify Your Bottleneck

Run diagnostic tests to understand what limits your rollout generation throughput.

:::{card}

**Task**: Identify whether parallelism settings, model server configuration, or verification logic are limiting throughput.

^^^

**This guide shows you how to**:

1. Run diagnostic tests with varying parallelism levels
2. Interpret throughput results
3. Determine next optimization steps

:::

---

## Diagnostic Test

Run the same collection task with different parallelism settings to see how throughput responds:

::::{tab-set}

:::{tab-item} Low Parallelism Baseline
```bash
time ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=test_100.jsonl \
    +output_jsonl_fpath=/tmp/test_p5.jsonl \
    +limit=100 \
    +num_samples_in_parallel=5
```

Watch the progress bar for `it/s` (iterations per second) metric:
```
Collecting rollouts: 100%|████| 100/100 [02:30<00:00, 0.67it/s]
```

Note the **total time** and **it/s** rate.
:::

:::{tab-item} Higher Parallelism Test
```bash
time ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=test_100.jsonl \
    +output_jsonl_fpath=/tmp/test_p20.jsonl \
    +limit=100 \
    +num_samples_in_parallel=20
```

Compare the new `it/s` rate:
```
Collecting rollouts: 100%|████| 100/100 [01:10<00:00, 1.43it/s]
```

If throughput **doubled or more** → parallelism was limiting factor  
If throughput **barely changed** → model server or verification is limiting factor
:::

::::

---

## Interpret Results

```{list-table}
:header-rows: 1
:widths: 40 60

* - Observation
  - Next Step
* - **Throughput increased significantly** (2x+)
  - {doc}`tune-parallelism` - Find your optimal parallelism value
* - **Throughput barely changed** (<20% improvement)
  - Check model server docs or simplify verification logic
* - **Errors or OOM crashes**
  - {doc}`tune-parallelism` - Reduce parallelism to stable level
```

---

## Next Step

Most users should proceed to {doc}`tune-parallelism` to find optimal concurrency settings.

:::{button-ref} tune-parallelism
:color: primary
:outline:
:ref-type: doc

Tune Parallelism →
:::

