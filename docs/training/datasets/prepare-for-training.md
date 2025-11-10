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
  - Familiar with how your resource server scores responses (see {doc}`../resource-servers/index`)
* - **Know your training algorithm**
  - SFT, DPO, or PPO—each needs different data preparation
* - **Understand reward patterns**
  - Binary (0/1), continuous (0.0-1.0), or multi-metric rewards
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

**Same-prompt pairing**: For best results, group rollouts by prompt and pair responses to the same prompt:

```python
from collections import defaultdict

by_prompt = defaultdict(list)
for r in rollouts:
    by_prompt[json.dumps(r['responses_create_params']['input'])].append(r)

# Pair best with worst from same prompt
pairs = []
for prompt_rollouts in by_prompt.values():
    prompt_rollouts.sort(key=lambda r: r['reward'], reverse=True)
    if len(prompt_rollouts) >= 2 and prompt_rollouts[0]['reward'] - prompt_rollouts[-1]['reward'] >= 0.2:
        pairs.append({'chosen': prompt_rollouts[0], 'rejected': prompt_rollouts[-1]})
```

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

**Tip**: If rewards are too binary (only 0.0 and 1.0), consider switching to a continuous reward server like `equivalence_llm_judge` for richer signal. See {doc}`../resource-servers/index` for options.

---

## Algorithm-Specific Guidance

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Algorithm
  - Preparation
  - Recommended Servers
* - **SFT**
  - Filter by high threshold (≥0.95)
  - mcqa, comp_coding, instruction_following
* - **DPO**
  - Create pairs with ≥0.2 gap, same-prompt preferred
  - library_judge_math, equivalence_llm_judge
* - **PPO/RL**
  - Use rollouts as-is, validate varied distribution
  - multineedle, library_judge_math
```


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

::::{tab-set}

:::{tab-item} Built-in Analysis
**Quick aggregated metrics**

```bash
# View aggregated metrics from rollouts
python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl
```

Shows averages, min, max for all numeric fields automatically.
:::

:::{tab-item} Custom Analysis Script
**Comprehensive rollout analysis**

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
:::

::::

---

## Next Steps

After preparing your data, integrate with your training framework:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL Integration
:link: ../integration/nemo-rl
:link-type: doc

Complete guide with transformation scripts for GRPO, SFT, and DPO
+++
{bdg-primary}`Recommended`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` All Frameworks
:link: ../integration/index
:link-type: doc

Integration guides for NeMo RL, VeRL, OpenRLHF, TRL
+++
{bdg-secondary}`Overview`
:::

::::

Or return to {doc}`index` for datasets overview.
