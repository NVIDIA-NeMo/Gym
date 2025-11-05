(training-data-quality-balancing)=

# Dataset Balancing

Balance task types and difficulty levels to prevent overfitting and improve generalization.

:::{card}

**Goal**: Ensure diverse, balanced training data that improves model generalization.

^^^

**You'll learn how to**:

1. Identify and fix task type imbalances
2. Balance difficulty levels for exploration
3. Apply stratified sampling strategies
4. Validate balanced distributions

:::

---

## Why Balance Datasets

**The problem**: Models overfit to frequent patterns in imbalanced data.

**Example**: If 80% of training data is math and 20% is coding, the model becomes a math specialist that performs poorly on coding.

**Solution**: Balance task types to ensure even representation.

---

## Task Type Balancing

### Identify Imbalances

```python
import json
from collections import Counter

def analyze_distribution(input_file):
    task_counts = Counter()
    total = 0
    
    for line in open(input_file):
        rollout = json.loads(line)
        task_type = rollout.get('metadata', {}).get('task_type', 'unknown')
        task_counts[task_type] += 1
        total += 1
    
    print("Task distribution:")
    for task_type, count in task_counts.most_common():
        print(f"  {task_type}: {count:,} ({count/total:.1%})")
    
    return task_counts

# Usage
analyze_distribution('filtered.jsonl')
```

**Output example**:
```
Task distribution:
  math: 8,472 (58.3%)
  coding: 3,214 (22.1%)
  search: 2,867 (19.7%)
```

**Red flag**: Any task type > 50% of dataset.

### Max-Per-Category Balancing

Cap each task type to N examples:

```python
from collections import defaultdict

def balance_by_task_type(input_file, output_file, max_per_type=1000):
    # Group by task type
    by_type = defaultdict(list)
    for line in open(input_file):
        rollout = json.loads(line)
        task_type = rollout.get('metadata', {}).get('task_type', 'general')
        by_type[task_type].append(line)
    
    # Cap each category
    with open(output_file, 'w') as out:
        for task_type, lines in by_type.items():
            selected = lines[:max_per_type]
            print(f"  {task_type}: {len(selected)} samples")
            for line in selected:
                out.write(line)
    
    total = sum(min(len(lines), max_per_type) for lines in by_type.values())
    print(f"\nBalanced dataset: {total} total samples")

# Usage
balance_by_task_type('filtered.jsonl', 'balanced.jsonl', max_per_type=2000)
```

**When to use**: One task type dominates (>50%)

**Trade-off**: May discard useful data from majority class

### Stratified Sampling

Maintain target proportions:

```python
import random

def stratified_sample(input_file, output_file, target_counts):
    """
    Sample specific number of examples per task type.
    
    Args:
        target_counts: Dict mapping task_type to desired count
                      e.g. {'math': 3000, 'coding': 3000, 'search': 3000}
    """
    # Group by task type
    by_type = defaultdict(list)
    for line in open(input_file):
        rollout = json.loads(line)
        task_type = rollout.get('metadata', {}).get('task_type', 'general')
        by_type[task_type].append(line)
    
    # Sample from each category
    with open(output_file, 'w') as out:
        for task_type, target_count in target_counts.items():
            available = by_type.get(task_type, [])
            
            if len(available) >= target_count:
                # Random sample if enough data
                selected = random.sample(available, target_count)
            else:
                # Take all if not enough
                selected = available
                print(f"⚠️ {task_type}: only {len(available)} available (wanted {target_count})")
            
            for line in selected:
                out.write(line)
            
            print(f"  {task_type}: {len(selected)} samples")

# Usage: Balance to 3K each
stratified_sample(
    'filtered.jsonl',
    'balanced.jsonl',
    target_counts={'math': 3000, 'coding': 3000, 'search': 3000}
)
```

**When to use**: Want specific target distribution

---

## Difficulty Balancing

Balance by reward score to ensure coverage of all difficulty levels.

### Define Difficulty Buckets

```python
def categorize_by_difficulty(rollout):
    reward = rollout.get('reward', 0)
    if reward >= 0.8:
        return 'easy'
    elif reward >= 0.4:
        return 'medium'
    else:
        return 'hard'
```

### Stratified Difficulty Sampling

```python
from collections import defaultdict
import random

def balance_by_difficulty(input_file, output_file, distribution=None):
    """
    Balance dataset by difficulty level.
    
    Args:
        distribution: Dict with target proportions, e.g.
                     {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}
    """
    if distribution is None:
        distribution = {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}
    
    # Group by difficulty
    by_difficulty = defaultdict(list)
    for line in open(input_file):
        rollout = json.loads(line)
        difficulty = categorize_by_difficulty(rollout)
        by_difficulty[difficulty].append(line)
    
    # Calculate total samples
    total_available = sum(len(lines) for lines in by_difficulty.values())
    
    # Sample according to distribution
    with open(output_file, 'w') as out:
        for difficulty, proportion in distribution.items():
            target_count = int(total_available * proportion)
            available = by_difficulty.get(difficulty, [])
            
            if len(available) >= target_count:
                selected = random.sample(available, target_count)
            else:
                selected = available
                print(f"⚠️ {difficulty}: only {len(available)} available (wanted {target_count})")
            
            for line in selected:
                out.write(line)
            
            print(f"  {difficulty}: {len(selected)} samples ({len(selected)/total_available:.1%})")

# Usage
balance_by_difficulty('filtered.jsonl', 'balanced.jsonl')
```

**Use for**:
- **RL training**: Ensure exploration across difficulty levels
- **Curriculum learning**: Create stratified subsets (easy → hard)

---

## Multi-Dimensional Balancing

Balance on both task type AND difficulty simultaneously.

```python
from collections import defaultdict
import random

def balance_multi_dimensional(input_file, output_file, samples_per_cell=500):
    """
    Balance by task type × difficulty.
    
    Args:
        samples_per_cell: Target samples for each (task_type, difficulty) combination
    """
    # Group by (task_type, difficulty)
    cells = defaultdict(list)
    for line in open(input_file):
        rollout = json.loads(line)
        task_type = rollout.get('metadata', {}).get('task_type', 'general')
        difficulty = categorize_by_difficulty(rollout)
        cells[(task_type, difficulty)].append(line)
    
    # Sample from each cell
    with open(output_file, 'w') as out:
        for (task_type, difficulty), lines in cells.items():
            if len(lines) >= samples_per_cell:
                selected = random.sample(lines, samples_per_cell)
            else:
                selected = lines
            
            for line in selected:
                out.write(line)
            
            print(f"  {task_type} × {difficulty}: {len(selected)} samples")

# Usage
balance_multi_dimensional('filtered.jsonl', 'balanced.jsonl', samples_per_cell=1000)
```

**When to use**: Need both task diversity AND difficulty balance (e.g., RL training)

---

## Balancing by Training Type

### SFT Balancing

Focus on task diversity (already high-quality from filtering).

```python
# Recommended: Max-per-category
balance_by_task_type('sft_filtered.jsonl', 'sft_balanced.jsonl', max_per_type=5000)
```

**Goal**: Prevent overfitting to common task types.

### DPO Balancing

Balance while maintaining preference pairs.

```python
def balance_dpo_pairs(input_file, output_file, max_pairs_per_type=1000):
    """Balance DPO pairs by task type."""
    by_type = defaultdict(list)
    
    for line in open(input_file):
        pair = json.loads(line)
        task_type = pair['chosen'].get('metadata', {}).get('task_type', 'general')
        by_type[task_type].append(line)
    
    with open(output_file, 'w') as out:
        for task_type, lines in by_type.items():
            selected = lines[:max_pairs_per_type]
            for line in selected:
                out.write(line)
            print(f"  {task_type}: {len(selected)} pairs")
```

**Key**: Keep pairs together (don't split chosen/rejected).

### RL Balancing

Balance task type AND difficulty for exploration.

```python
# Recommended: Multi-dimensional balancing
balance_multi_dimensional('rl_filtered.jsonl', 'rl_balanced.jsonl', samples_per_cell=800)
```

**Goal**: Ensure coverage of state space (task × difficulty).

---

## Validation After Balancing

Check that distribution matches targets:

```python
def validate_balance(balanced_file):
    # Check task distribution
    task_counts = Counter()
    difficulty_counts = Counter()
    
    for line in open(balanced_file):
        rollout = json.loads(line)
        task_type = rollout.get('metadata', {}).get('task_type', 'unknown')
        difficulty = categorize_by_difficulty(rollout)
        
        task_counts[task_type] += 1
        difficulty_counts[difficulty] += 1
    
    total = sum(task_counts.values())
    
    print("Task type distribution:")
    for task_type, count in task_counts.most_common():
        print(f"  {task_type}: {count} ({count/total:.1%})")
    
    print("\nDifficulty distribution:")
    for difficulty, count in difficulty_counts.most_common():
        print(f"  {difficulty}: {count} ({count/total:.1%})")

# Usage
validate_balance('balanced.jsonl')
```

**Success criteria**:
- No single task type > 40%
- Difficulty distribution matches targets (if specified)

---

## Balancing Workflow Template

Complete pipeline:

```python
import json
import random
from collections import defaultdict, Counter

def balance_dataset(input_file, output_file, strategy='task_type', **kwargs):
    """
    Complete balancing workflow.
    
    Args:
        strategy: 'task_type', 'difficulty', or 'multi'
        **kwargs: Strategy-specific parameters
    """
    print(f"Balancing with strategy: {strategy}")
    print(f"\nBefore balancing:")
    analyze_distribution(input_file)
    
    # Apply strategy
    if strategy == 'task_type':
        max_per_type = kwargs.get('max_per_type', 2000)
        balance_by_task_type(input_file, output_file, max_per_type)
    
    elif strategy == 'difficulty':
        distribution = kwargs.get('distribution')
        balance_by_difficulty(input_file, output_file, distribution)
    
    elif strategy == 'multi':
        samples_per_cell = kwargs.get('samples_per_cell', 500)
        balance_multi_dimensional(input_file, output_file, samples_per_cell)
    
    print(f"\nAfter balancing:")
    validate_balance(output_file)

# Usage examples
balance_dataset('filtered.jsonl', 'balanced.jsonl', strategy='task_type', max_per_type=3000)
balance_dataset('filtered.jsonl', 'balanced.jsonl', strategy='difficulty')
balance_dataset('filtered.jsonl', 'balanced.jsonl', strategy='multi', samples_per_cell=1000)
```

---

## Common Pitfalls

### Over-Balancing

**Symptom**: Dataset shrinks too much (< 50% retention)

**Cause**: Caps too aggressive or not enough minority class samples

**Solution**: 
- Increase max_per_type
- Use weighted sampling instead of hard caps

### Insufficient Minority Data

**Symptom**: Can't meet balance targets for some categories

**Causes**:
- Minority classes filtered too aggressively
- Need more rollouts from minority tasks

**Solutions**:
- Collect more data from underrepresented tasks
- Relax filtering for minority classes
- Use weighted sampling/reweighting

### Loss of Valuable Data

**Symptom**: Discarding high-quality examples from majority class

**Solution**: Save discarded data for future use:
```python
# Save overflow to separate file
overflow = lines[max_per_type:]
with open(f'{task_type}_overflow.jsonl', 'w') as f:
    for line in overflow:
        f.write(line)
```

---

## Next Steps

**Format for training** → {doc}`../datasets/prepare-for-training`

**Validate format** → Use `ng_prepare_data` for final validation

**Train** → Pass balanced dataset to training framework
