(training-data-quality)=

# Data Quality

Validate training data quality using NeMo Gym's automatic metrics before passing to RL frameworks.

:::{card}

**Goal**: Quickly assess if rollouts are ready for training.

^^^

**You'll learn how to**:

1. Interpret NeMo Gym's automatic metric aggregation
2. Validate quality by RL algorithm type (PPO, DPO)
3. Identify broken verification before training

:::

---

## Automatic Metrics

NeMo Gym automatically aggregates metrics after collection. Refer to {doc}`../about/concepts/rollout-collection-fundamentals` for detailed explanation of how automatic metric aggregation works.

**Quick summary**: Any numeric field from your resource server's `verify()` is averaged across all rollouts.

```bash
ng_collect_rollouts +input_jsonl_fpath=tasks.jsonl +output_jsonl_fpath=rollouts.jsonl

# Displays after collection:
# {
#   "reward": 0.73,
#   "accuracy": 0.68,
#   "avg_tool_calls": 2.1
# }
```

**Built-in analysis**:

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl
```

---

## Quick Quality Checks

### Reward Distribution

```python
import json

rewards = [json.loads(line)['reward'] for line in open('rollouts.jsonl')]

print(f"Count: {len(rewards)}")
print(f"Mean: {sum(rewards)/len(rewards):.2f}")
print(f"Min: {min(rewards):.2f}")
print(f"Max: {max(rewards):.2f}")

# Red flags
if min(rewards) == max(rewards):
    print("⚠️ All rewards identical - verification may be broken")
if sum(rewards)/len(rewards) < 0.3:
    print("⚠️ Very low average - tasks may be too hard")
```

### Success Rate

```python
successes = sum(1 for line in open('rollouts.jsonl') 
                if json.loads(line).get('success', False))
total = sum(1 for _ in open('rollouts.jsonl'))

print(f"Success rate: {successes/total:.1%}")
```

---

## Quality by RL Algorithm

Different algorithms have different data requirements.

### PPO (Proximal Policy Optimization)

**Needs**: Diverse quality range for on-policy learning

**Expected metrics**:
- Reward distribution: 0.3-0.9 (balanced)
- Success rate: 30-70%
- Quality signal: Spread, not clustered

**Red flags**:
- All high rewards (>0.9): No learning signal
- All low rewards (<0.3): May need easier tasks

### DPO (Direct Preference Optimization)

**Needs**: Pairs with clear quality difference

**Expected metrics**:
- Reward spread: Both high (>0.7) and low (<0.5)
- Success rate: 40-70% (need failures for rejected samples)
- Pairs generated: Multiple samples per task

**Validation**:
```python
rewards = [json.loads(line)['reward'] for line in open('rollouts.jsonl')]
high = sum(1 for r in rewards if r > 0.7)
low = sum(1 for r in rewards if r < 0.5)

print(f"High quality: {high} ({high/len(rewards):.1%})")
print(f"Low quality: {low} ({low/len(rewards):.1%})")

if high < 0.2 * len(rewards) or low < 0.2 * len(rewards):
    print("⚠️ Insufficient quality spread for DPO")
```

**Red flags**:
- All high rewards: No rejected samples for preferences
- All similar rewards: No clear preference signal

---

## Integration Checkpoints

Before passing rollouts to RL frameworks:

**Volume**:
- Minimum: 1K rollouts for quick iteration
- Typical: 10K-100K for training
- Scale: 1M+ for production runs

**Quality**:
- Mean reward reasonable for your task (not all 0 or all 1)
- Distribution matches RL algorithm needs (spread for PPO, diverse for DPO)
- No obvious verification failures (see {doc}`../rollout-collection/optimize-for-training/production-scale` quick checks)

**Format**:
- Use `ng_prepare_data` to validate JSONL format before training
- Refer to {doc}`../datasets/validate-format` for format requirements

---

## Custom Filtering and Balancing

**NeMo Gym does not provide built-in filtering or balancing utilities.**

If your RL framework or training objectives require custom filtering or balancing:

1. **Implement based on your RL framework's requirements**
   - Each framework (VeRL, NeMo-RL, OpenRLHF, TRL) has different preferences
   - Refer to your framework's documentation for recommended preprocessing

2. **Use NeMo Gym metrics to guide thresholds**
   - Automatic metrics help identify appropriate quality thresholds
   - Apply filtering based on your training objectives

3. **Leverage framework-native preprocessing**
   - Many RL frameworks include data filtering and balancing
   - Use framework capabilities rather than custom pipelines

**Example filtering patterns** shown in {doc}`../tutorials/offline-training-w-rollouts`.

---

## Next Steps

**Validate format** → {doc}`../datasets/validate-format` before training

**Integrate with RL framework** → {doc}`../integration/index` for framework-specific guides

**Monitor during collection** → {doc}`../rollout-collection/optimize-for-training/production-scale` for real-time tracking
