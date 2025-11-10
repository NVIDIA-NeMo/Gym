(training-rollout-optimize-production)=

# Training Throughput at Scale

Maximize training data generation using NeMo Gym's built-in throughput features and verification optimization strategies.

:::{card}

**Goal**: Generate 100K-1M+ training rollouts efficiently using NeMo Gym configuration and optimization features.

^^^

**This guide shows you how to**:

1. Monitor throughput with NeMo Gym's progress tracking
2. Resume interrupted runs using automatic append mode
3. Optimize verification logic for training speed
4. Track and validate data quality at scale

:::

---

## NeMo Gym Features

Built-in capabilities for production-scale collection.

### Progress Monitoring

NeMo Gym automatically displays a real-time progress bar when you run `ng_collect_rollouts`:

```bash
ng_collect_rollouts \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl

# Automatically displays:
# Collecting rollouts: 45%|████▌     | 450/1000 [02:15<02:45, 3.33it/s]
```

**Key metric**: `it/s` (items per second) = throughput in samples per second

Use this to spot bottlenecks in real time—if throughput drops or stalls, see {ref}`Verification Optimization <verification-optimization>` below.

### Automatic Metrics

NeMo Gym automatically displays aggregated metrics after collection completes:

```json
{
  "reward": 0.73,
  "accuracy": 0.68,
  "avg_tool_calls": 2.1
}
```

Any numeric field from your resource server's verification is automatically averaged.

Refer to {ref}`concepts-rc-fundamentals` for detailed explanation of how automatic metric aggregation works.

### Resume Interrupted Runs

NeMo Gym automatically opens output files in **append mode**, so interrupted runs can be resumed without losing progress:

```bash
# Check how many completed
wc -l rollouts.jsonl  # Output: 5432

# Process remaining tasks
tail -n +5433 input.jsonl > remaining.jsonl
ng_collect_rollouts \
    +input_jsonl_fpath=remaining.jsonl \
    +output_jsonl_fpath=rollouts.jsonl  # Appends automatically
    +limit=4568
```

### Parameter Overrides

Override model parameters globally via CLI:

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +responses_create_params.max_output_tokens=512
```

:::{dropdown} How Overrides Work
:icon: code

```python
row["responses_create_params"] = row["responses_create_params"] | config.responses_create_params
```

CLI overrides merge with per-task parameters. CLI takes precedence.
:::

---

(verification-optimization)=

## Verification Optimization

If verification is slow, it bottlenecks collection throughput.

### Detecting the Bottleneck

```{list-table}
:header-rows: 1
:widths: 50 50

* - Symptom
  - Likely Cause
* - Model responds fast, but `it/s` is slow
  - Verification taking too long
* - High CPU usage during collection
  - Compute-heavy verification
* - Progress bar stalls between samples
  - External API calls in verification
```

### Optimization Patterns

::::{tab-set}

:::{tab-item} Cache Lookups

```python
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

```python
def verify(self, task, response):
    if self.config.get('fast_mode', False):
        return quick_heuristic(response)
    else:
        return precise_verification(response)
```

Use fast mode for training, precise for evaluation.

:::

:::{tab-item} Defer Verification

1. Return placeholder reward from resource server
2. Collect at full speed without verification
3. Run separate verification pass later

**Trade-off**: No real-time quality feedback

:::

::::

---

## Throughput Tracking

Measure and analyze collection performance for training optimization.

### Samples Per Second

Monitor real-time throughput directly from the progress bar:

```bash
ng_collect_rollouts \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl

# Output shows:
# Collecting rollouts: 45%|████▌     | 450/1000 [02:15<02:45, 3.33it/s]
#                                                           ^^^^^ samples/sec
```

**Interpreting throughput**:

- **< 1 it/s**: Model server or verification bottleneck, tune parallelism
- **1-5 it/s**: Normal for complex verification or moderate parallelism
- **> 5 it/s**: Good throughput, I/O bound or well-optimized

### Tokens Per Second

For more precise cost and performance analysis:

```python
import json
import time

# Measure collection time
start = time.time()
# Run ng_collect_rollouts here
elapsed = time.time() - start

# Calculate token throughput
total_tokens = sum(
    json.loads(line).get('usage', {}).get('total_tokens', 0)
    for line in open('rollouts.jsonl')
)
tokens_per_sec = total_tokens / elapsed

print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")
print(f"Total tokens: {total_tokens:,}")
```

**Use tokens/sec for**:

- Comparing different model server configurations
- Estimating training dataset generation time
- Cost analysis when using paid API endpoints

### Benchmarking Collections

Test throughput with small samples before full runs:

```bash
# Quick benchmark (100 samples)
time ng_collect_rollouts \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=test_rollouts.jsonl \
    +limit=100

# Calculate throughput
# If 100 samples took 30 seconds: 100 / 30 = 3.33 samples/sec
# Estimate for 100K: 100,000 / 3.33 = ~8.3 hours
```

---

## Quick Quality Checks

Catch obvious data issues during collection with fast smoke tests.

:::{tip}
**These are quick sanity checks**, not comprehensive quality analysis. For full filtering, curation, and balancing strategies, see {doc}`../../data-quality/index`.
:::

### Spot Check Metrics

Use NeMo Gym's automatic metrics (see after collection completes) as a quick sanity check:

**Red flags**:
- **All rewards identical** (0.0 or 1.0): Verification logic may be broken
- **Accuracy far from baseline**: Verification criteria may be too strict/loose
- **Unexpected values**: Review resource server implementation

Refer to {doc}`../../data-quality/index` for comprehensive quality validation.

### Quick Reward Distribution Check

Run a fast distribution check to catch broken verification:

```python
import json

# Load rewards
rewards = [
    json.loads(line)['reward']
    for line in open('rollouts.jsonl')
]

# Quick distribution summary
print(f"Min: {min(rewards):.2f}")
print(f"Max: {max(rewards):.2f}")
print(f"Mean: {sum(rewards)/len(rewards):.2f}")
print(f"Median: {sorted(rewards)[len(rewards)//2]:.2f}")

# Spot obvious issues
if min(rewards) == max(rewards):
    print("⚠️ Warning: All rewards identical, check verification logic")
```

**What to look for**:

- **No variation** (all same value): Broken verification logic
- **All zeros or all ones**: Verification may not be running properly
- **Extreme outliers**: Check for bugs in reward calculation

:::{note}
This is a quick smoke test. For comprehensive quality analysis, reward thresholds, and filtering strategies, see {doc}`../../data-quality/index`.
:::

### Detect Obviously Broken Rollouts

Scan for critical issues that indicate collection problems:

```python
import json

issues = []
for i, line in enumerate(open('rollouts.jsonl')):
    data = json.loads(line)
    
    # Empty responses (model/agent failure)
    if not data.get('response', {}).get('content'):
        issues.append(f"Line {i}: Empty response")
    
    # Error responses (agent crashed)
    if data.get('response', {}).get('content') in ['ERROR', 'FAILED']:
        issues.append(f"Line {i}: Error response")
    
    # Extreme reward outliers (verification bug)
    if data.get('reward', 0) < -10 or data.get('reward', 0) > 10:
        issues.append(f"Line {i}: Unusual reward {data['reward']}")

if issues:
    print(f"⚠️ Found {len(issues)} critical issues:")
    for issue in issues[:10]:  # Show first 10
        print(f"  {issue}")
    print("\nThese indicate collection problems. Fix before proceeding to quality filtering.")
```

**When to use this**: After collection completes, before investing in comprehensive quality analysis.

---

## Next Steps

### Continue Optimizing

**Tune sampling** → {doc}`../sampling-strategies/index` for temperature and diversity  
**Reference patterns** → {doc}`../collection-patterns/index` for copy-paste commands

### Move to Quality Phase

After collection completes and quick checks pass:

**Data quality analysis and filtering** → {doc}`../../data-quality/index`

:::{seealso}
**Pipeline flow**: Optimize throughput (this guide) → Collect rollouts → Quick checks (above) → Comprehensive quality analysis ({doc}`../../data-quality/index`) → Format for training ({doc}`../../datasets/index`)
:::</seealso>
