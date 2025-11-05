(training-data-quality-filtering)=

# Filtering Strategies

Remove low-quality rollouts using threshold-based filtering tailored to your training objective.

:::{card}

**Goal**: Curate high-quality training data by filtering rollouts that don't meet quality criteria.

^^^

**You'll learn how to**:

1. Apply reward and success thresholds
2. Filter by training type (SFT, DPO, RL)
3. Handle edge cases (empty responses, errors, outliers)
4. Validate filtering results

:::

:::{tip}
**Quality over quantity**: Training on 5K high-quality examples beats training on 50K noisy ones.
:::

---

## Basic Filtering Patterns

### Reward Threshold

```python
import json

def filter_by_reward(input_file, output_file, min_reward=0.8):
    with open(input_file) as f, open(output_file, 'w') as out:
        kept = 0
        total = 0
        
        for line in f:
            rollout = json.loads(line)
            total += 1
            
            if rollout.get('reward', 0) >= min_reward:
                out.write(line)
                kept += 1
        
        print(f"Kept {kept}/{total} ({kept/total:.1%})")

# Usage
filter_by_reward('rollouts.jsonl', 'filtered.jsonl', min_reward=0.8)
```

**Choosing thresholds**:
- **SFT**: 0.8-0.9 (strict, high-quality only)
- **RL**: 0.3-0.4 (permissive, keep exploration range)
- **DPO**: See pair filtering below

### Success Field

```python
def filter_by_success(input_file, output_file):
    with open(input_file) as f, open(output_file, 'w') as out:
        kept = 0
        total = 0
        
        for line in f:
            rollout = json.loads(line)
            total += 1
            
            if rollout.get('success', False):
                out.write(line)
                kept += 1
        
        print(f"Kept {kept}/{total} ({kept/total:.1%})")
```

**When to use**: Tasks with clear pass/fail (code execution, exact match).

### Length Bounds

```python
def filter_by_length(input_file, output_file, min_turns=2, max_turns=20):
    with open(input_file) as f, open(output_file, 'w') as out:
        kept = 0
        total = 0
        
        for line in f:
            rollout = json.loads(line)
            total += 1
            
            num_turns = len(rollout.get('output', []))
            if min_turns <= num_turns <= max_turns:
                out.write(line)
                kept += 1
        
        print(f"Kept {kept}/{total} ({kept/total:.1%})")
```

**Use to remove**:
- Trivial single-turn responses (min_turns)
- Degenerate loops (max_turns)

---

## Composite Filtering

Combine multiple criteria for robust filtering.

### Multi-Criterion Filter

```python
def filter_rollouts(input_file, output_file, criteria):
    with open(input_file) as f, open(output_file, 'w') as out:
        kept = 0
        total = 0
        
        for line in f:
            rollout = json.loads(line)
            total += 1
            
            # Check all criteria
            passes = True
            
            if rollout.get('reward', 0) < criteria['min_reward']:
                passes = False
            
            if criteria.get('require_success') and not rollout.get('success', False):
                passes = False
            
            num_turns = len(rollout.get('output', []))
            if not (criteria['min_turns'] <= num_turns <= criteria['max_turns']):
                passes = False
            
            # Check for error responses
            response_content = rollout.get('response', {}).get('content', '')
            if response_content in ['ERROR', 'FAILED', '']:
                passes = False
            
            if passes:
                out.write(line)
                kept += 1
        
        print(f"Kept {kept}/{total} ({kept/total:.1%})")
        return kept, total
```

---

## Training-Type Specific Filtering

### SFT: Conservative Filtering

Keep only demonstration-quality examples.

```python
def filter_for_sft(input_file, output_file):
    criteria = {
        'min_reward': 0.8,           # High quality only
        'require_success': True,      # Must pass verification
        'min_turns': 2,               # Non-trivial interactions
        'max_turns': 15               # Reasonable length
    }
    
    kept, total = filter_rollouts(input_file, output_file, criteria)
    
    # Expected: 60-80% retention
    if kept / total < 0.5:
        print("⚠️ Warning: Low retention rate. Consider relaxing thresholds.")
```

**Expected retention**: 60-80%

### DPO: Preference Pair Filtering

Generate pairs with clear quality difference.

```python
from collections import defaultdict

def filter_for_dpo(input_file, output_file, min_gap=0.1):
    # Group by task
    by_task = defaultdict(list)
    for line in open(input_file):
        rollout = json.loads(line)
        task_id = rollout.get('task_id') or rollout.get('input')
        by_task[task_id].append(rollout)
    
    # Generate pairs
    pairs = []
    for task_id, samples in by_task.items():
        if len(samples) < 2:
            continue
        
        # Sort by reward
        sorted_samples = sorted(samples, key=lambda x: x.get('reward', 0), reverse=True)
        
        # Create pairs with sufficient gap
        for i in range(len(sorted_samples) - 1):
            chosen = sorted_samples[i]
            rejected = sorted_samples[i + 1]
            
            reward_gap = chosen.get('reward', 0) - rejected.get('reward', 0)
            if reward_gap >= min_gap:
                pairs.append({'chosen': chosen, 'rejected': rejected})
    
    # Write pairs
    with open(output_file, 'w') as out:
        for pair in pairs:
            out.write(json.dumps(pair) + '\n')
    
    print(f"Generated {len(pairs)} preference pairs from {sum(len(s) for s in by_task.values())} rollouts")
    return pairs

# Usage
filter_for_dpo('rollouts.jsonl', 'dpo_pairs.jsonl', min_gap=0.15)
```

**Key points**:
- Requires 2+ samples per task (use `num_repeats` during collection)
- Quality gap should be clear (0.1-0.2 minimum)
- Both responses must be valid (no errors)

### RL: Permissive Filtering

Keep diverse quality range for exploration.

```python
def filter_for_rl(input_file, output_file):
    criteria = {
        'min_reward': 0.3,            # Permissive threshold
        'require_success': False,     # Keep failures for learning
        'min_turns': 1,               # Allow short interactions
        'max_turns': 25               # Remove only extreme outliers
    }
    
    kept, total = filter_rollouts(input_file, output_file, criteria)
    
    # Expected: 85-95% retention
    if kept / total < 0.8:
        print("⚠️ Warning: Over-filtering for RL. Relax thresholds.")
```

**Expected retention**: 85-95%

---

## Advanced Filtering

### Domain-Specific Filtering

**Code Execution**:
```python
def filter_code_execution(input_file, output_file):
    with open(input_file) as f, open(output_file, 'w') as out:
        for line in f:
            rollout = json.loads(line)
            
            # Must have tool call result
            has_execution = any(
                item.get('type') == 'function_call_output'
                for item in rollout.get('output', [])
            )
            
            if has_execution and rollout.get('reward', 0) >= 0.5:
                out.write(line)
```

**Math Reasoning**:
```python
def filter_math_reasoning(input_file, output_file):
    with open(input_file) as f, open(output_file, 'w') as out:
        for line in f:
            rollout = json.loads(line)
            
            # Must show reasoning
            has_reasoning = any(
                item.get('type') == 'reasoning'
                for item in rollout.get('output', [])
            )
            
            if has_reasoning and rollout.get('reward', 0) >= 0.7:
                out.write(line)
```

### Deduplication

Remove exact duplicates:

```python
def deduplicate(input_file, output_file):
    seen = set()
    with open(input_file) as f, open(output_file, 'w') as out:
        for line in f:
            rollout = json.loads(line)
            
            # Create hash of response content
            response_text = str(rollout.get('response', {}).get('content', ''))
            response_hash = hash(response_text)
            
            if response_hash not in seen:
                seen.add(response_hash)
                out.write(line)
    
    print(f"Kept {len(seen)} unique rollouts")
```

---

## Filtering Workflow Template

Complete end-to-end script:

```python
import json
from collections import defaultdict

def curate_training_data(input_file, output_file, training_type='sft'):
    """
    Complete filtering workflow.
    
    Args:
        input_file: Raw rollouts JSONL
        output_file: Filtered rollouts JSONL
        training_type: 'sft', 'dpo', or 'rl'
    """
    print(f"Filtering for {training_type.upper()} training...")
    
    # Apply appropriate filter
    if training_type == 'sft':
        filter_for_sft(input_file, output_file)
    elif training_type == 'dpo':
        filter_for_dpo(input_file, output_file)
    elif training_type == 'rl':
        filter_for_rl(input_file, output_file)
    else:
        raise ValueError(f"Unknown training type: {training_type}")
    
    # Validate results
    print("\nValidation:")
    rewards = [
        json.loads(line).get('reward', 0)
        for line in open(output_file)
    ]
    print(f"  Mean reward: {sum(rewards)/len(rewards):.2f}")
    print(f"  Min reward: {min(rewards):.2f}")
    print(f"  Max reward: {max(rewards):.2f}")

# Usage
curate_training_data('rollouts.jsonl', 'filtered_sft.jsonl', training_type='sft')
```

---

## Validation After Filtering

Run quality-metrics analysis on filtered data:

```python
# Quick validation
filtered_rewards = [
    json.loads(line)['reward']
    for line in open('filtered.jsonl')
]

print(f"Filtered dataset: {len(filtered_rewards)} samples")
print(f"Mean reward: {sum(filtered_rewards)/len(filtered_rewards):.2f}")
print(f"Min reward: {min(filtered_rewards):.2f}")

# Should match training type expectations
```

**For SFT**: Mean reward should be ≥ 0.8

**For RL**: Reward distribution should still be spread (not all high)

**For DPO**: Check pair quality gaps

---

## Common Pitfalls

### Over-Filtering

**Symptom**: Retention rate < 50%

**Causes**:
- Thresholds too strict
- Tasks genuinely difficult (not a filtering issue)

**Solutions**:
- Lower threshold incrementally (0.8 → 0.7 → 0.6)
- Check if verification is overly strict
- Review sample of filtered-out examples

### Under-Filtering

**Symptom**: Many low-quality examples remain

**Causes**:
- Thresholds too permissive
- Missing composite filters (length, errors)

**Solutions**:
- Add error detection filter
- Combine multiple criteria
- Manual spot-check filtered data

### Data Loss

**Symptom**: Dataset too small after filtering

**Solutions**:
- Generate more rollouts before filtering
- Use weighted sampling instead of hard filtering
- Consider if task difficulty matches capability

---

## Next Steps

**Balance filtered data** → {doc}`dataset-balancing` for task diversity

**Format for training** → {doc}`../datasets/prepare-for-training`

**Validate format** → Use `ng_prepare_data` before training
