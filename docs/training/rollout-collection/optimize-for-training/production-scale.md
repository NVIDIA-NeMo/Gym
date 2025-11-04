(training-rollout-optimize-production)=

# Production Scale

Monitor throughput, handle interruptions, and distribute workloads for large-scale rollout generation.

:::{card}

**Task**: Apply production patterns to scale rollout generation: resume interrupted runs, distribute across machines, and optimize verification.

^^^

**This guide shows you how to**:

1. Monitor throughput with built-in progress metrics
2. Resume interrupted collections using append mode
3. Distribute processing with chunked datasets
4. Optimize verification for high-throughput scenarios

:::

---

## Monitor and Measure

Track metrics to validate improvements and detect regressions.

### Real-Time Monitoring

During collection, watch the progress bar:

```
Collecting rollouts: 45%|████▌     | 450/1000 [02:15<02:45, 3.33it/s]
```

**Key metric**: `it/s` (iterations per second) = samples per second

### Computing Throughput

**Samples per second**:
```bash
# Time your collection
time ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=benchmark.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +limit=1000

# Calculate: 1000 samples / [total seconds] = samples/sec
```

**Tokens per second** (more precise):
```python
import json

total_tokens = 0
with open('rollouts.jsonl') as f:
    for line in f:
        rollout = json.loads(line)
        # Sum input + output tokens if available
        total_tokens += rollout.get('usage', {}).get('total_tokens', 0)

elapsed_seconds = 300  # From `time` command
print(f"Tokens/sec: {total_tokens / elapsed_seconds:.2f}")
```

### Aggregate Metrics

After collection completes, NeMo Gym automatically displays aggregated metrics:

```json
{
  "reward": 0.73,
  "accuracy": 0.68,
  "avg_tool_calls": 2.1
}
```

```{dropdown} How Metric Aggregation Works
:icon: gear

From `rollout_collection.py:103-109`:
```python
metrics.update({k: v for k, v in result.items() if isinstance(v, (int, float))})

# After collection:
avg_metrics = {k: v / len(rows) for k, v in metrics.items()}
print(json.dumps(avg_metrics, indent=4))
```

Any numeric field returned by verification is automatically averaged across all rollouts.
```

### Tracking Over Time

Create a simple log for trend analysis:

```bash
# Append metrics to log file
echo "$(date),1000,$elapsed_sec,$samples_per_sec,$avg_reward" >> throughput_log.csv

# Later, analyze trends
column -t -s, throughput_log.csv
```

---

## Production Patterns

Strategies for large-scale generation in production environments.

### Incremental Collection

Append to existing files to resume interrupted jobs:

```bash
# Initial run (may be interrupted)
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=large_dataset.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +limit=10000

# If interrupted, check how many completed
wc -l rollouts.jsonl
# Output: 5432 rollouts.jsonl

# Resume from where you left off
tail -n +5433 large_dataset.jsonl > remaining.jsonl
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=remaining.jsonl \
    +output_jsonl_fpath=rollouts.jsonl  # Appends due to 'a' mode
    +limit=4568  # 10000 - 5432
```

```{dropdown} Why This Works
:icon: code

From `rollout_collection.py:94`:
```python
with open(config.output_jsonl_fpath, "a") as f:
```

Output file opens in **append mode** by default, so you can safely resume by processing remaining tasks.
```

### Chunked Processing

Split large datasets for parallel processing across machines:

```bash
# Split 100K dataset into 10 chunks
split -l 10000 -d --additional-suffix=.jsonl huge_dataset.jsonl chunk_

# Distribute chunks to different machines/GPUs
# Machine 1:
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=chunk_00.jsonl \
    +output_jsonl_fpath=rollouts_00.jsonl

# Machine 2:
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=chunk_01.jsonl \
    +output_jsonl_fpath=rollouts_01.jsonl

# ... continue for all chunks ...

# Merge results
cat rollouts_*.jsonl > final_rollouts.jsonl
```

### Continuous Generation

For long-running jobs, use tmux or screen for resilience:

```bash
# Start detachable session
tmux new -s rollout_collection

# Inside tmux, run collection
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=million_tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl

# Detach: Ctrl+b, then d
# Reattach later: tmux attach -t rollout_collection
```

Monitor with:
```bash
# Check progress
watch -n 60 'wc -l rollouts.jsonl'

# Monitor GPU usage
watch -n 5 nvidia-smi
```

---

## Verification Optimization

Verification runs during collection—if it's slow, it becomes a bottleneck.

### Detecting Verification Bottleneck

```{list-table}
:header-rows: 1
:widths: 50 50

* - Sign
  - Interpretation
* - Model responds fast, but `it/s` is slow
  - Verification likely bottleneck
* - High CPU usage during collection
  - Compute-heavy verification
* - Progress bar stalls between samples
  - Verification waiting on external call
```

### Optimization Approaches

::::{tab-set}

:::{tab-item} Cache Lookups
If verification repeats expensive operations:

```python
# In your resource server's verify() function
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_lookup(key):
    return result

def verify(self, task, response):
    data = expensive_lookup(task['key'])
    return compute_reward(data, response)
```
:::

:::{tab-item} Fast Mode
Implement approximate verification for training data:

```python
def verify(self, task, response):
    if self.config.get('fast_mode', False):
        return quick_heuristic(response)
    else:
        return precise_verification(response)
```

Use fast mode for training collection, precise mode for evaluation.
:::

:::{tab-item} Defer Verification
Collect rollouts without verification, verify in batch later:

1. Modify resource server to return placeholder reward
2. Collect at full speed
3. Run separate verification pass with higher parallelism

**Trade-off**: No real-time quality feedback during collection
:::

::::

---

## Parameter Overrides

NeMo Gym allows overriding model parameters via CLI to reduce latency:

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +responses_create_params.max_output_tokens=512
```

**Why this helps**: Shorter outputs = faster generation = higher throughput

```{dropdown} How Parameter Override Works
:icon: code

From `rollout_collection.py:97`:
```python
row["responses_create_params"] = row["responses_create_params"] | config.responses_create_params
```

CLI overrides merge with per-task parameters. CLI values take precedence.
```

---

## Next Steps

Now that you understand production-scale patterns:

**Tune data characteristics** → {doc}`../sampling-strategies/index` for temperature and sampling strategies  
**See complete patterns** → {doc}`../collection-patterns/index` for copy-paste commands

