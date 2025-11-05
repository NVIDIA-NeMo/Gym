(training-datasets-prepare-for-training)=

# Prepare for Training

Transform rollouts with reward scores into training-ready datasets for SFT, DPO, or PPO.

Convert `rollouts.jsonl` with reward scores into the specific format required by your training algorithm.

## Before You Start

```{list-table}
:header-rows: 1
:widths: 30 70

* - Prerequisite
  - Description
* - **Collected rollouts**
  - You have `rollouts.jsonl` with reward scores from {doc}`../rollout-collection/index`
* - **Understand reward signals**
  - Familiar with how your resource server scores responses (see {doc}`../verification/index`)
* - **Know your training algorithm**
  - SFT, DPO, or PPO—each needs different data preparation
* - **Understand reward patterns**
  - Binary (0/1), continuous (0.0-1.0), or multi-metric rewards (see {doc}`../verification/reward-shaping`)
```

**Related concepts**:
- {doc}`index` - Overview of dataset formats and pipeline
- {doc}`../data-quality/index` - For filtering and curation before preparation
- {doc}`../verification/multi-objective-scoring` - For handling multi-metric rollouts

---

## Preparing for SFT

**Goal**: Filter for high-quality examples only.

### Basic Pattern: Threshold Filtering

```python
import json

# Load rollouts
rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Keep only high-reward examples
sft_data = [r for r in rollouts if r['reward'] >= 0.95]

print(f"Filtered: {len(sft_data)}/{len(rollouts)} examples ({100*len(sft_data)/len(rollouts):.1f}%)")

# Write SFT dataset
with open('sft_data.jsonl', 'w') as f:
    for item in sft_data:
        f.write(json.dumps(item) + '\n')
```

**Threshold guidelines**:
- **Binary servers**: Use `reward == 1.0` (exact match)
- **Continuous servers**: Use `reward >= 0.90` or `reward >= 0.95`
- **Adjust based on data volume**: Lower threshold if you need more data

### Multi-Criteria Filtering

For servers with multiple metrics, filter by multiple conditions:

```python
# Example with multineedle (has reward, accuracy, set_overlap)
sft_data = [r for r in rollouts 
            if r['reward'] == 1.0          # Must be fully correct
            and r['set_overlap'] >= 0.95]  # And nearly complete
```

### Quality Validation

Before training, validate your filtered data:

```python
import statistics

rewards = [r['reward'] for r in sft_data]

print(f"SFT Dataset Size: {len(sft_data)}")
print(f"Mean Reward: {statistics.mean(rewards):.3f}")
print(f"Min Reward: {min(rewards):.3f}")

# Spot-check high and low examples
print("\n--- Sample High Reward ---")
print(sft_data[0]['response']['output'][-1])
```

**Red flags**:
- Too few examples (< 100): Lower threshold or collect more rollouts
- All rewards identical: Verification may be too strict
- Low mean reward (< 0.90): Quality threshold may be too low

---

## Preparing for DPO

**Goal**: Create preference pairs with clear quality differences.

### Basic Pattern: Stratified Pairing

```python
import json

rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Sort by reward
sorted_rollouts = sorted(rollouts, key=lambda r: r['reward'])

# Define quality tiers
high_quality = [r for r in rollouts if r['reward'] >= 0.7]
low_quality = [r for r in rollouts if r['reward'] <= 0.4]

# Create pairs with clear separation
pairs = []
for chosen in high_quality:
    for rejected in low_quality:
        gap = chosen['reward'] - rejected['reward']
        if gap >= 0.2:  # Minimum quality gap
            pairs.append({
                'chosen': chosen,
                'rejected': rejected,
                'reward_gap': gap
            })

print(f"Created {len(pairs)} DPO pairs")
print(f"Mean reward gap: {sum(p['reward_gap'] for p in pairs) / len(pairs):.3f}")

# Write DPO dataset
with open('dpo_pairs.jsonl', 'w') as f:
    for pair in pairs:
        f.write(json.dumps(pair) + '\n')
```

**Quality gap guidelines**:
- **Minimum gap**: 0.2 (20% difference)
- **Preferred gap**: 0.3–0.5 (clear quality difference)
- **Avoid**: Pairs with gap < 0.1 (ambiguous preference)

### Same-Prompt Pairing

For better DPO training, pair responses to the same prompt:

```python
from collections import defaultdict

# Group by prompt
by_prompt = defaultdict(list)
for rollout in rollouts:
    prompt = rollout['responses_create_params']['input']
    prompt_key = json.dumps(prompt)  # Use serialized prompt as key
    by_prompt[prompt_key].append(rollout)

# Create pairs within each prompt
pairs = []
for prompt_key, prompt_rollouts in by_prompt.items():
    # Sort by reward
    prompt_rollouts.sort(key=lambda r: r['reward'], reverse=True)
    
    # Pair best with worst from same prompt
    if len(prompt_rollouts) >= 2:
        for i in range(len(prompt_rollouts) // 2):
            chosen = prompt_rollouts[i]
            rejected = prompt_rollouts[-(i+1)]
            gap = chosen['reward'] - rejected['reward']
            
            if gap >= 0.2:
                pairs.append({
                    'chosen': chosen,
                    'rejected': rejected,
                    'reward_gap': gap
                })

print(f"Same-prompt pairs: {len(pairs)}")
```

### Validation

```python
import statistics

gaps = [p['reward_gap'] for p in pairs]

print(f"DPO Dataset: {len(pairs)} pairs")
print(f"Mean gap: {statistics.mean(gaps):.3f}")
print(f"Min gap: {min(gaps):.3f}")
print(f"Max gap: {max(gaps):.3f}")

# Check distribution
print(f"Gaps >= 0.3: {sum(1 for g in gaps if g >= 0.3)}/{len(gaps)}")
```

**Red flags**:
- Mean gap < 0.2: Pairs too similar
- Many gaps < 0.1: Insufficient separation
- All gaps near 1.0: May be too easy (binary rewards only)

---

## Preparing for PPO

**Goal**: Use continuous rewards directly, validate distribution.

### Basic Pattern: Distribution Validation

```python
import json
import statistics

rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# PPO uses rewards as-is from rollouts
rewards = [r['reward'] for r in rollouts]

# Analyze distribution
print(f"PPO Dataset: {len(rollouts)} rollouts")
print(f"Mean: {statistics.mean(rewards):.3f}")
print(f"Std Dev: {statistics.stdev(rewards):.3f}")
print(f"Min: {min(rewards):.3f}")
print(f"Max: {max(rewards):.3f}")

# Check for clustering
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for i in range(len(bins)-1):
    count = sum(1 for r in rewards if bins[i] <= r < bins[i+1])
    print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} ({100*count/len(rewards):.1f}%)")
```

**Good PPO distribution characteristics**:
- **Varied**: Rewards spread across range (not clustered at 0.0 or 1.0)
- **Standard deviation**: > 0.15 (indicates discrimination)
- **Coverage**: Multiple bins populated

**Red flags**:
- All rewards near 0.0 or 1.0: Task may need adjustment
- Std dev < 0.1: Rewards not discriminative enough
- Single peak at 0.5: Verification may be noisy

### Advanced: Reward Shaping

If distribution is too binary, consider intermediate resource server:

```python
# Binary server (mcqa) might give:
# [0.0, 0.0, 1.0, 0.0, 1.0, 1.0] - only two values

# Consider using continuous server (equivalence_llm_judge) for:
# [0.12, 0.35, 0.88, 0.41, 0.92, 0.87] - rich signal
```

See {doc}`../verification/index` for resource server selection guidance.

---

## Handling Different Training Algorithms

### Quick Reference

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Algorithm
  - Reward Needs
  - Recommended Servers
* - **SFT**
  - Binary or high-threshold continuous
  - mcqa, comp_coding, instruction_following
* - **DPO**
  - Continuous with clear separation (≥0.2 gap)
  - library_judge_math, equivalence_llm_judge
* - **PPO/RL**
  - Continuous with rich signal (varied distribution)
  - multineedle, library_judge_math
```

### Algorithm-Specific Tips

::::{tab-set}

:::{tab-item} SFT
**Focus**: High-quality examples only

**Preparation**:
- Filter by high threshold (≥0.95)
- Validate all kept examples are correct
- Volume matters less than quality

**Code**:
```python
sft_data = [r for r in rollouts if r['reward'] >= 0.95]
```
:::

:::{tab-item} DPO
**Focus**: Clear preference signals

**Preparation**:
- Ensure reward gap ≥0.2 between chosen/rejected
- Prefer same-prompt pairs when possible
- Balance dataset (equal chosen/rejected distribution)

**Code**:
```python
# Same-prompt pairing with gap check
pairs = create_pairs(rollouts, min_gap=0.2, same_prompt=True)
```
:::

:::{tab-item} PPO
**Focus**: Rich continuous signal

**Preparation**:
- Use rewards directly from rollouts
- Validate varied distribution (not clustered)
- Monitor std dev (should be > 0.15)

**Code**:
```python
# PPO uses rollouts as-is
# Just validate distribution
assert statistics.stdev(rewards) > 0.15
```
:::

::::

---

## Multi-Metric Rollouts

If your resource server returns multiple metrics, you can filter or pair by different criteria.

### Example: Multineedle

```json
{
  "reward": 0.82,
  "accuracy": 0.82,
  "set_overlap": 0.95
}
```

### Filtering Strategies

**Strict (exact correctness)**:
```python
# SFT: Only fully correct examples
sft_data = [r for r in rollouts if r['accuracy'] == 1.0]
```

**Lenient (partial credit)**:
```python
# SFT: High partial credit OK
sft_data = [r for r in rollouts if r['set_overlap'] >= 0.90]
```

**Multi-criteria**:
```python
# DPO: High accuracy AND high overlap
high_quality = [r for r in rollouts 
                if r['accuracy'] >= 0.8 and r['set_overlap'] >= 0.9]

# DPO: Low on both
low_quality = [r for r in rollouts 
               if r['accuracy'] <= 0.3 and r['set_overlap'] <= 0.4]

pairs = create_pairs(high_quality, low_quality, min_gap=0.3)
```

See {doc}`../verification/multi-objective-scoring` for detailed multi-metric strategies.

---

## Tools and Scripts

### Built-in Analysis

```bash
# View aggregated metrics from rollouts
python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl
```

Shows averages, min, max for all numeric fields automatically.

### Custom Analysis Script

```python
import json
import statistics

def analyze_rollouts(jsonl_path):
    """Comprehensive rollout analysis"""
    rollouts = [json.loads(line) for line in open(jsonl_path)]
    
    # Basic stats
    rewards = [r['reward'] for r in rollouts]
    print(f"Dataset: {len(rollouts)} rollouts")
    print(f"Rewards: mean={statistics.mean(rewards):.3f}, "
          f"std={statistics.stdev(rewards):.3f}")
    
    # Distribution
    print("\nReward Distribution:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for r in rewards if bins[i] <= r < bins[i+1])
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} "
              f"({'█' * (count // 10)})")
    
    # Quality estimates
    print(f"\nSFT candidates (reward≥0.95): {sum(1 for r in rewards if r >= 0.95)}")
    print(f"DPO high (reward≥0.7): {sum(1 for r in rewards if r >= 0.7)}")
    print(f"DPO low (reward≤0.4): {sum(1 for r in rewards if r <= 0.4)}")
    
    return rollouts

# Usage
rollouts = analyze_rollouts('rollouts.jsonl')
```

---

## Integration with Training Frameworks

After preparing data, integrate with your chosen RL framework:

### VeRL

See {doc}`../integration/verl` for VeRL-specific data format.

### NeMo-RL

See {doc}`../integration/nemo-rl` for NeMo-RL integration.

### OpenRLHF

See {doc}`../integration/openrlhf` for OpenRLHF data preparation.

### TRL

See {doc}`../integration/trl` for TRL (Transformers Reinforcement Learning) format.

---

## Troubleshooting

:::{dropdown} Not Enough High-Quality Examples for SFT
**Problem**: After filtering, too few examples remain

**Solutions**:
- Lower threshold (0.95 → 0.90)
- Collect more rollouts
- Check if verification is too strict
- Use different resource server
:::

:::{dropdown} Insufficient Reward Separation for DPO
**Problem**: Can't find pairs with gap ≥0.2

**Solutions**:
- Use continuous reward server instead of binary
- Collect more diverse rollouts
- Lower minimum gap requirement (0.15)
- Collect more rollouts per prompt
:::

:::{dropdown} Reward Distribution Too Narrow for PPO
**Problem**: All rewards cluster at 0.5

**Solutions**:
- Check verification logic (may be noisy)
- Use different resource server
- Adjust task difficulty
- Verify resource server configuration
:::

:::{dropdown} Multi-Metric Confusion
**Problem**: Multiple metrics give conflicting signals

**Solutions**:
- Use `reward` field (primary training signal)
- Understand what each metric measures
- Filter by primary metric first
- See {doc}`../verification/multi-objective-scoring`
:::

---

## Best Practices

**Before large-scale collection**:
- [ ] Collect small sample (100 rollouts)
- [ ] Analyze reward distribution
- [ ] Test filtering/pairing with sample
- [ ] Validate prepared data quality
- [ ] Scale to full collection

**For production training**:
- [ ] Version your rollouts (`rollouts_v1.jsonl`, `rollouts_v2.jsonl`)
- [ ] Keep preparation scripts in version control
- [ ] Log preparation parameters (thresholds, gaps)
- [ ] Validate prepared data before training
- [ ] Monitor metrics during training

**Quality checks**:
- [ ] Spot-check high-reward examples (look good?)
- [ ] Spot-check low-reward examples (look bad?)
- [ ] Verify pairs have clear quality difference
- [ ] Ensure distribution matches algorithm needs

---

## Next Steps

:::{button-ref} ../integration/index
:color: primary
:outline:
:ref-type: doc

Integrate with Training Frameworks →
:::

Or return to {doc}`index` for datasets overview.
