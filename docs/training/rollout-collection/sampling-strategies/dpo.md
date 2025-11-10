(training-rollout-sampling-dpo)=

# DPO Sampling Strategy

Configure for Direct Preference Optimization: generating diverse preference pairs with quality differences.

:::{card}

**Task**: Generate preference pairs for DPO training with multiple diverse responses per prompt ranked by quality.

^^^

**This guide shows you how to**:

1. Configure temperature and repeats for behavioral diversity
2. Balance parallelism to maintain quality variation
3. Group and rank rollouts into preference pairs
4. Filter and format data for DPO training

:::

---

## Before You Start

Ensure you have these prerequisites before generating DPO data:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Get Started completed**
  - Complete {doc}`../../../get-started/collecting-rollouts` first
* - **Servers running**
  - Agent and model servers with verification that returns reward/score
* - **Training objective**
  - Understanding of DPO (learning from pairwise preference comparisons)
* - **Task dataset**
  - Input prompts in JSONL format (5K+ recommended for DPO)
* - **Verification quality**
  - Verification must produce reliable quality scores for ranking
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Collect DPO Data

Generate multiple diverse responses per prompt to create preference pairs for DPO training.

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=dpo_prompts.jsonl \
    +output_jsonl_fpath=dpo_rollouts.jsonl \
    +responses_create_params.temperature=<temperature> \
    +responses_create_params.top_p=<top_p> \
    +num_repeats=<num_repeats> \
    +num_samples_in_parallel=<parallelism>
```

**Configuration**: For DPO, use higher temperature for diversity, multiple repeats (3-4) to generate comparisons, and moderate parallelism. Refer to {doc}`parameters` for parameter explanations.

### Expected Output

```text
Found 5000 rows!
Repeating rows (in a pattern of abc to aabbcc) from 5000 to 15000!
Querying with 10 concurrent requests
Collecting rollouts: 100%|████████████| 15000/15000 [15:23<00:00, 16.24it/s]
{
    "reward": 0.612,
    "accuracy": 0.558
}
```

**Note**: Average reward moderate (not peaked) due to diversity strategy.

---

## Create Preference Pairs

**NeMo Gym collects rollouts—you create the pairs.** The collection guarantees every `num_repeats` consecutive rollouts share the same input, making grouping straightforward:

```python
import json

# Load rollouts
with open('dpo_rollouts.jsonl') as f:
    rollouts = [json.loads(line) for line in f]

# Group: every 3 consecutive rollouts = 1 group
num_repeats = 3
groups = [rollouts[i:i+num_repeats] for i in range(0, len(rollouts), num_repeats)]

# Create pairs from each group
pairs = []
for group in groups:
    # Sort by reward
    sorted_group = sorted(group, key=lambda x: x['reward'], reverse=True)
    
    # Skip if reward gap too small
    if sorted_group[0]['reward'] - sorted_group[-1]['reward'] < 0.15:
        continue
    
    # Create pair
    pairs.append({
        'prompt': sorted_group[0]['responses_create_params']['input'],
        'chosen': sorted_group[0]['output'],
        'rejected': sorted_group[-1]['output'],
        'reward_gap': sorted_group[0]['reward'] - sorted_group[-1]['reward']
    })

print(f"Created {len(pairs)} preference pairs from {len(groups)} groups")
```

Check reward distributions to validate quality:

```bash
# Check chosen vs rejected spread
jq '.reward' dpo_rollouts.jsonl | sort -n | uniq -c
```

---

## Troubleshooting

::::{tab-set}

:::{tab-item} Not Enough Diversity

**Problem**: Few rollouts differ enough to make quality pairs

```bash
# Increase temperature for more behavioral variance
ng_collect_rollouts ... +responses_create_params.temperature=0.8

# Or collect more attempts per prompt
ng_collect_rollouts ... +num_repeats=4
```

:::

:::{tab-item} Responses Too Erratic

**Problem**: Nonsensical or off-task outputs

```bash
# Lower temperature
ng_collect_rollouts ... +responses_create_params.temperature=0.6

# Or filter out low-quality rollouts before pairing
jq 'select(.reward >= 0.3)' dpo_rollouts.jsonl > dpo_filtered.jsonl
```

:::

::::
