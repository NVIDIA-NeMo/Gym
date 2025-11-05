(training-rollout-optimize-parallelism)=

# Tune Parallelism

Find the optimal `num_samples_in_parallel` value that maximizes throughput for your setup.

:::{card}

**Task**: Systematically test different parallelism values to find the setting that maximizes throughput without overwhelming your infrastructure.

^^^

**This guide shows you how to**:

1. Understand how NeMo Gym limits concurrent requests
2. Find optimal parallelism systematically
3. Validate stability under load

:::

:::{seealso}
Not sure if parallelism is your bottleneck? Start with {doc}`identify-bottleneck` to diagnose what's limiting throughput.
:::

---

## How Parallelism Works

NeMo Gym uses `asyncio.Semaphore` to limit concurrent requests:

```python
semaphore = nullcontext()
if config.num_samples_in_parallel:
    semaphore = Semaphore(config.num_samples_in_parallel)

async with semaphore:
    response = await server_client.post("/run", task)
    save_rollout(response)
```

**Behavior**:
- **When set**: Limits to N concurrent requests
- **When omitted**: No limit (all tasks submitted concurrently)

**Why this matters**: Setting parallelism prevents overwhelming your model server or hitting rate limits while maximizing throughput.

---

## Finding Optimal Value

Use a systematic doubling approach to find your throughput ceiling.

### 1. Establish Baseline

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tuning_dataset.jsonl \
    +output_jsonl_fpath=/tmp/baseline.jsonl \
    +limit=200 \
    +num_samples_in_parallel=5
```

Watch the progress bar for throughput:
```
Collecting rollouts: 100%|████| 200/200 [02:00<00:00, 1.67it/s]
```

**Record the result**: 1.67 samples/sec

### 2. Double and Measure

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tuning_dataset.jsonl \
    +output_jsonl_fpath=/tmp/test_p10.jsonl \
    +limit=200 \
    +num_samples_in_parallel=10
```

**Expect**: ~3.0 samples/sec (if parallelism was limiting)

### 3. Continue Until Plateau

Keep doubling (20, 40, 80) until:

- **Throughput plateaus** → Found your ceiling
- **Throughput degrades** → Exceeded capacity, back off
- **Errors appear** → Reduce parallelism

**Optimal value**: Use 80% of peak to leave headroom for stability.

### 4. Validate Stability

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=larger_dataset.jsonl \
    +output_jsonl_fpath=/tmp/stability_test.jsonl \
    +limit=1000 \
    +num_samples_in_parallel=[your_optimal_value]
```

Run a longer test to ensure stability at scale

---

## Default Behavior

When `num_samples_in_parallel` is **not specified**, NeMo Gym submits **all tasks concurrently** with no limit. This can:

- Overwhelm model servers
- Exceed API rate limits  
- Cause OOM errors

Always set this parameter for production workloads.

---

## Next Steps

After finding optimal parallelism:

**Production patterns** → {doc}`production-scale` for monitoring and scale strategies  
**Sampling strategies** → {doc}`../sampling-strategies/index` for temperature and diversity tuning

