(training-data-quality-metrics)=

# Quality Metrics

Track and analyze training data quality using automatic metrics and distribution analysis.

:::{card}

**Goal**: Understand your data quality before filtering and training.

^^^

**You'll learn how to**:

1. Read NeMo Gym's automatic metric summaries
2. Analyze reward distributions by training type
3. Compute success rates and identify issues
4. Track task diversity and detect imbalances

:::

---

## Automatic Metrics

NeMo Gym automatically aggregates metrics after collection completes.

### What Gets Tracked

Any numeric field returned by your resource server's `verify()` function is automatically averaged:

```python
# In your resource server
def verify(self, task, response):
    return {
        "reward": 0.85,           # ← tracked
        "accuracy": 1.0,          # ← tracked  
        "avg_tool_calls": 3,      # ← tracked
        "execution_time_ms": 247  # ← tracked
    }
```

After collection:

```json
{
  "reward": 0.73,
  "accuracy": 0.68,
  "avg_tool_calls": 2.1,
  "execution_time_ms": 189.3
}
```

### Quick Interpretation

Use automatic metrics for immediate quality assessment:

- **reward < 0.3**: Verification may be too strict or tasks too hard
- **reward > 0.95**: Verification may be too lenient or tasks too easy
- **All metrics identical**: Verification broken (see production-scale quick checks)

---

## Reward Distribution Analysis

Essential for detecting broken verification and choosing filter thresholds.

### Compute Distribution

```python
import json

# Load rewards
rewards = [json.loads(line)['reward'] for line in open('rollouts.jsonl')]

# Distribution stats
print(f"Count: {len(rewards)}")
print(f"Min: {min(rewards):.2f}")
print(f"Max: {max(rewards):.2f}")  
print(f"Mean: {sum(rewards)/len(rewards):.2f}")
print(f"Median: {sorted(rewards)[len(rewards)//2]:.2f}")

# Percentiles
sorted_rewards = sorted(rewards)
p25 = sorted_rewards[len(rewards)//4]
p75 = sorted_rewards[3*len(rewards)//4]
print(f"25th percentile: {p25:.2f}")
print(f"75th percentile: {p75:.2f}")
```

### Expected Patterns by Training Type

**SFT (Supervised Fine-Tuning)**:
- Most rewards > 0.8 (high-quality demonstrations)
- Small tail of failures (0.0-0.4)
- If mean < 0.7: tasks too hard or need better prompts

**DPO (Direct Preference Optimization)**:
- Spread across range (0.2-1.0)
- Need both high and low for preference pairs
- If all > 0.8: increase temperature or diversity

**RL (Reinforcement Learning)**:
- Balanced distribution (0.3-0.9)
- Shows exploration across quality levels
- If clustered: policy may have collapsed

### Visualize Distribution

```python
import matplotlib.pyplot as plt

plt.hist(rewards, bins=20, edgecolor='black')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.title('Reward Distribution')
plt.axvline(0.8, color='r', linestyle='--', label='SFT threshold')
plt.legend()
plt.savefig('reward_distribution.png')
```

---

## Success Rate Analysis

Percentage of rollouts passing verification.

### Compute Success Rate

```python
import json

successes = sum(
    1 for line in open('rollouts.jsonl')
    if json.loads(line).get('success', False)
)
total = sum(1 for _ in open('rollouts.jsonl'))

success_rate = successes / total
print(f"Success rate: {success_rate:.1%} ({successes}/{total})")
```

### Target Rates by Training Type

- **SFT**: ≥80% (strict quality requirement)
- **DPO**: 40-70% (need both successes and failures)
- **RL**: 30-70% (exploration range)

**Warning signs**:
- **< 20%**: Verification too strict or tasks too hard
- **> 95%**: Verification too lenient or tasks too easy

---

## Length Distribution

Detect degenerate responses (too short, too long, or stuck at max length).

### Analyze Response Lengths

```python
import json

lengths = [
    len(json.loads(line).get('output', []))
    for line in open('rollouts.jsonl')
]

print(f"Min turns: {min(lengths)}")
print(f"Max turns: {max(lengths)}")
print(f"Avg turns: {sum(lengths)/len(lengths):.1f}")

# Check for max-length responses (indicates cutoff)
max_length_count = sum(1 for l in lengths if l >= 20)
if max_length_count > len(lengths) * 0.1:
    print(f"⚠️ Warning: {max_length_count} rollouts hit max length (10%+)")
```

**Red flags**:
- Many single-turn responses: Agent not engaging properly
- Many max-length responses: Degenerate loops or cutoff issues
- Extreme outliers: Check specific examples

---

## Task Diversity

Ensure balanced representation across task types.

### Analyze Task Distribution

```python
import json
from collections import Counter

task_counts = Counter()
for line in open('rollouts.jsonl'):
    data = json.loads(line)
    task_type = data.get('metadata', {}).get('task_type', 'unknown')
    task_counts[task_type] += 1

total = sum(task_counts.values())
print("Task distribution:")
for task_type, count in task_counts.most_common():
    print(f"  {task_type}: {count} ({count/total:.1%})")
```

**Warning signs**:
- One task type > 50%: Consider balancing before training
- Missing task types: Check input data coverage

---

## Aggregate Statistics Script

Use NeMo Gym's built-in script for quick summaries:

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl
```

Output:
```json
{
  "reward": 0.734,
  "accuracy": 0.689,
  "avg_tool_calls": 2.3
}
```

---

## Quality Validation Checklist

Before proceeding to filtering, verify:

**For SFT**:
- [ ] Mean reward ≥ 0.7
- [ ] Success rate ≥ 80%
- [ ] No single task type > 60%

**For DPO**:
- [ ] Reward spread across range (not all high)
- [ ] Success rate 40-70%
- [ ] Multiple samples per task

**For RL**:
- [ ] Balanced reward distribution
- [ ] Success rate 30-70%
- [ ] Diverse task representation

---

## Next Steps

**Proceed to filtering** → {doc}`filtering-strategies` to apply quality thresholds

**Balance dataset** → {doc}`dataset-balancing` to ensure task diversity

**Format for training** → {doc}`../datasets/prepare-for-training` after quality curation
