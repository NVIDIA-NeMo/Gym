(tutorial-offline-training)=

# Offline Training with Rollouts (SFT/DPO)

You've generated rollouts in the get-started series—now learn how to transform them into training data that can improve your AI models through supervised fine-tuning (SFT) and direct preference optimization (DPO).

:::{card}

**Goal**: Transform generated rollouts into high-quality training data for supervised fine-tuning and preference optimization.

^^^

**In this tutorial, you will**:

1. Understand SFT and DPO training data formats
2. Filter rollouts for quality
3. Process rollouts into training formats
4. Validate and evaluate your training data

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New to NeMo Gym? Start with Get Started
:::

:::{tip}
**Prerequisites**: This tutorial assumes you've completed the [Get Started](../get-started/index.md) series and understand how to [collect rollouts](../get-started/collecting-rollouts.md). If you need a deeper understanding of rollout collection strategies, refer to [Rollout Collection Fundamentals](../about/concepts/rollout-collection-fundamentals.md).
:::

---

## Why Offline Training?

**Offline training** uses pre-collected rollouts to improve AI models without real-time exploration. This approach is ideal when:

- You have a working agent that demonstrates good behaviors
- You want reproducible results - same data, consistent training outcomes
- You need cost-effective training - no expensive exploration during training
- You want to capture expert demonstrations - preserve successful patterns
- You have limited compute - more efficient than reinforcement learning

**The offline training pipeline**: Generate rollouts → Filter and process → Train models → Deploy improved agents

---

## Training Data Types

Choose the training approach that matches your goal:

::::{tab-set}

:::{tab-item} SFT (Supervised Fine-Tuning)
:sync: sft

**Purpose**: Train models to follow successful agent interaction patterns

**When to use**:
- You have high-quality examples of desired behavior
- Want the model to imitate specific interaction patterns
- Need consistent, predictable responses

**Data structure**: Input-output pairs showing complete agent conversations

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "content": "Temperature: 22°C, sunny"},
    {"role": "assistant", "content": "The weather in Paris is 22°C and sunny."}
  ],
  "quality_score": 0.95
}
```

**Collection strategy**: Low temperature (0.1-0.3), single rollout per task

:::

:::{tab-item} DPO (Direct Preference Optimization)
:sync: dpo

**Purpose**: Train models to prefer better responses over worse ones

**When to use**:
- You can generate multiple responses with varying quality
- Want to teach the model to distinguish good from bad
- Need to improve response quality through comparison

**Data structure**: Preference pairs with chosen vs rejected responses

```json
{
  "prompt": [{"role": "user", "content": "Solve this math problem: 2x + 5 = 13"}],
  "chosen": [
    {"role": "assistant", "content": "I'll solve for x step by step:\n2x + 5 = 13\n2x = 13 - 5\n2x = 8\nx = 4"}
  ],
  "rejected": [
    {"role": "assistant", "content": "The answer is x = 3"}
  ],
  "quality_difference": 0.7
}
```

**Collection strategy**: Higher temperature (0.6-0.8), 2+ rollouts per task for comparison

:::

::::

---

## Data Preparation Overview

The offline training pipeline follows this logical flow:

1. **Collect rollouts** using strategies from the {doc}`../get-started/collecting-rollouts` guide and {doc}`../about/concepts/rollout-collection-fundamentals` reference
2. **Filter for quality** - Remove poor rollouts before processing
3. **Format for training** - Convert to SFT or DPO format based on your goals

### Quick Reference: SFT vs DPO Data Requirements

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Aspect
  - SFT
  - DPO
* - **Rollouts per task**
  - 1 rollout per task
  - 2+ rollouts per task
* - **Temperature**
  - Low (0.1-0.3)
  - Higher (0.6-0.8)
* - **Quality filter**
  - Keep only high-quality (reward ≥ 0.8)
  - Keep pairs with quality difference ≥ 0.1
* - **Output format**
  - Conversation history
  - Preference pairs (chosen vs rejected)
* - **Best for**
  - Imitating expert behavior
  - Learning preferences between responses
```

---

## 1. Quality Filtering and Curation

Always filter your rollouts first before formatting them for training. Here are example approaches you can customize for your needs:

### Automatic Filtering

```python
def filter_rollouts(input_file: str, output_file: str, filters: Dict):
    """Apply automatic quality filters to rollouts."""
    with open(input_file) as f, open(output_file, 'w') as out:
        kept = 0
        total = 0
        
        for line in f:
            rollout = json.loads(line)
            total += 1
            
            # Apply filters
            if (rollout.get('reward', 0) >= filters.get('min_reward', 0.5) and
                rollout.get('success', False) and
                len(rollout.get('output', [])) >= filters.get('min_turns', 2) and
                len(rollout.get('output', [])) <= filters.get('max_turns', 20)):
                
                out.write(line)
                kept += 1
        
        print(f"Kept {kept}/{total} rollouts ({kept/total*100:.1f}%)")

# Apply filters first
filter_rollouts('raw_rollouts.jsonl', 'filtered_rollouts.jsonl', {
    'min_reward': 0.7,
    'min_turns': 3,
    'max_turns': 15
})
```

**✅ Success Check**: You should see output like `Kept 847/1203 rollouts (70.4%)` showing how many rollouts passed your quality filters.

### Manual Curation (Optional)

For critical applications, sample and manually review:

```python
def sample_for_review(input_file: str, sample_size: int = 50):
    """Sample rollouts for manual review."""
    import random
    
    with open(input_file) as f:
        rollouts = [json.loads(line) for line in f]
    
    # Stratified sampling by reward
    low_reward = [r for r in rollouts if r.get('reward', 0) < 0.5]
    mid_reward = [r for r in rollouts if 0.5 <= r.get('reward', 0) < 0.8]
    high_reward = [r for r in rollouts if r.get('reward', 0) >= 0.8]
    
    sample = (random.sample(low_reward, min(10, len(low_reward))) +
              random.sample(mid_reward, min(20, len(mid_reward))) +
              random.sample(high_reward, min(20, len(high_reward))))
    
    with open('manual_review_sample.jsonl', 'w') as out:
        for rollout in sample:
            out.write(json.dumps(rollout) + '\n')
```

:::{note}
These are example filtering approaches. Customize the criteria, thresholds, and sampling strategies based on your specific domain and quality requirements.
:::

---

## 2. Format for Training

Once you have filtered, high-quality rollouts, format them for your chosen training method:

::::{tab-set}

:::{tab-item} SFT Data Processing
:sync: sft

Transform filtered rollouts into conversation format:

```python
import json
from typing import List, Dict

def process_sft_data(filtered_rollout_file: str, output_file: str):
    """Convert filtered rollouts to SFT training format."""
    with open(filtered_rollout_file) as f, open(output_file, 'w') as out:
        for line in f:
            rollout = json.loads(line)
            sft_example = {
                "messages": rollout['output'],
                "reward": rollout['reward'],
                "task_type": rollout.get('metadata', {}).get('task_type', 'general')
            }
            out.write(json.dumps(sft_example) + '\n')

# Process filtered rollouts (no additional filtering needed)
process_sft_data('filtered_rollouts.jsonl', 'sft_data.jsonl')
```

**✅ Success Check**: Your `sft_data.jsonl` file should now contain one training example per rollout, each with the conversation history and reward score.

:::

:::{tab-item} DPO Data Processing
:sync: dpo

Create preference pairs from filtered rollouts (requires 2 rollouts per task):

```python
def create_dpo_pairs(filtered_rollout_file: str, output_file: str):
    """Create preference pairs from pairs of filtered rollouts."""
    
    # Group rollouts by task
    task_groups = {}
    with open(filtered_rollout_file) as f:
        for line in f:
            rollout = json.loads(line)
            task_id = rollout.get('task_id') or hash(json.dumps(rollout['responses_create_params']['input']))
            
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(rollout)
    
    # Create preference pairs from pairs of rollouts
    pairs = []
    for task_rollouts in task_groups.values():
        if len(task_rollouts) == 2:  # DPO works with pairs
            rollout_1, rollout_2 = task_rollouts
            
            # Determine which is better based on reward
            if rollout_1['reward'] > rollout_2['reward']:
                chosen, rejected = rollout_1, rollout_2
            else:
                chosen, rejected = rollout_2, rollout_1
            
            # Only create pair if there's meaningful difference
            quality_diff = chosen['reward'] - rejected['reward']
            if quality_diff >= 0.1:  # Minimum difference threshold
                pairs.append({
                    "prompt": chosen['responses_create_params']['input'],
                    "chosen": chosen['output'],
                    "rejected": rejected['output'],
                    "quality_difference": quality_diff
                })
    
    # Save preference pairs
    with open(output_file, 'w') as out:
        for pair in pairs:
            out.write(json.dumps(pair) + '\n')
    
    print(f"Created {len(pairs)} preference pairs")

# Create DPO pairs from filtered rollouts
create_dpo_pairs('filtered_rollouts.jsonl', 'dpo_pairs.jsonl')
```

**✅ Success Check**: You should see output like `Created 423 preference pairs` showing how many comparison pairs were generated from your rollouts.

:::

::::

---

## Training Integration

Once you have your processed data (`sft_data.jsonl` or `dpo_pairs.jsonl`), you can use any post-training framework for SFT or DPO:

### Standard Data Formats

SFT data follows the conversation format used by most training libraries:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

DPO data follows the preference pair format:

```json
{"prompt": ["..."], "chosen": ["..."], "rejected": ["..."]}
```

---

## Validation and Evaluation

### Pre-Training Validation

Before training, validate your data quality by checking:

- **Dataset size**: Sufficient examples for training objectives
- **Reward distribution**: Reasonable range and average quality scores  
- **Length distribution**: Appropriate conversation lengths
- **Task diversity**: Balanced representation across different task types

### Post-Training Evaluation

Test your improved model by generating new rollouts on held-out evaluation tasks:

```bash
# Generate rollouts with improved model
ng_collect_rollouts +agent_name=improved_agent \
    +input_jsonl_fpath=evaluation_tasks.jsonl \
    +output_jsonl_fpath=post_training_rollouts.jsonl
```

Compare key metrics like average reward, success rate, and task-specific performance against your baseline to measure improvement.

---

## Best Practices

:::{dropdown} 1. Data Quality Over Quantity
:icon: shield-check

**Principle**: Better to train on fewer high-quality examples than many noisy ones.

**Implementation**:

```python
# Prefer high-quality filtered data over large noisy datasets
filter_criteria = {
    'min_reward': 0.8,        # High threshold for SFT
    'min_success_rate': 0.9,
    'require_tool_usage': True  # Domain-specific requirements
}
```

**Why it matters**: Training on low-quality data can teach the model bad behaviors that are difficult to unlearn.

:::

:::{dropdown} 2. Balanced Datasets
:icon: rows

**Principle**: Ensure diverse task representation to prevent overfitting to common patterns.

**Implementation**:

```python
# Ensure diverse task representation
def balance_dataset(input_file: str, output_file: str, max_per_category: int = 100):
    task_counts = {}
    balanced_data = []
    
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            task_type = data.get('metadata', {}).get('task_type', 'general')
            
            if task_counts.get(task_type, 0) < max_per_category:
                balanced_data.append(data)
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    with open(output_file, 'w') as out:
        for data in balanced_data:
            out.write(json.dumps(data) + '\n')
```

**Why it matters**: Imbalanced datasets lead to models that perform well on common tasks but fail on less frequent ones.

:::

:::{dropdown} 3. Iterative Improvement
:icon: iterations

**Principle**: Use a continuous improvement cycle to progressively enhance agent quality.

**The cycle**:

1. Generate rollouts with current agent
2. Filter and prepare training data  
3. Train improved model
4. Deploy and evaluate
5. Use improved agent to generate better rollouts

**Why it matters**: Each iteration builds on previous improvements, creating a virtuous cycle of enhancement.

:::

:::{dropdown} 4. Version Control
:icon: versions

**Principle**: Track data and model versions for reproducibility and debugging.

**Implementation**:

```bash
# Track data versions
mkdir -p training/v1.0/
mv sft_data.jsonl training/v1.0/
mv dpo_pairs.jsonl training/v1.0/

# Track model versions  
mkdir -p models/agent_v1.0/
cp -r ./results/* models/agent_v1.0/
```

**Why it matters**: Version control enables you to reproduce results, compare different approaches, and roll back if needed.

:::


---

## Troubleshooting

:::{dropdown} Problem: Poor Training Data Quality
:icon: alert
:color: warning

**Symptoms:**
- Low average rewards (< 0.5)
- Inconsistent behaviors across similar tasks
- Model produces irrelevant or incorrect responses after training

**Root causes:**
- Base agent needs improvement
- Filtering thresholds too permissive
- Verification function not aligned with quality goals

**Solutions:**

1. **Tighten filters**: Increase `min_reward` threshold from 0.7 to 0.8 or higher
2. **Lower temperature**: Generate rollouts with temperature 0.1-0.3 for more consistent behavior
3. **Manual curation**: Sample and review rollouts before training
4. **Improve base agent**: Fix obvious issues before generating training data

:::

:::{dropdown} Problem: Insufficient Data Diversity
:icon: git-branch
:color: warning

**Symptoms:**
- Model performs well on training tasks but fails on variations
- Overfitting to specific patterns or phrasings
- Poor generalization to new task types

**Root causes:**
- Input dataset too narrow
- Temperature too low during generation
- Imbalanced task distribution

**Solutions:**

1. **Increase diversity**: Generate rollouts with higher temperature (0.6-0.8)
2. **Expand input tasks**: Use more diverse prompts and task variations
3. **Multiple configurations**: Collect data from different agent setups or system prompts
4. **Balance dataset**: Use the balancing script to ensure even task representation

:::

:::{dropdown} Problem: Training Instability
:icon: flame
:color: danger

**Symptoms:**
- Loss doesn't converge or oscillates wildly
- Model performance degrades after training
- Gradient explosions or NaN values

**Root causes:**
- Data format incompatibility
- Extreme outliers in conversation length
- Learning rate too high

**Solutions:**

1. **Verify format**: Check that data matches your training framework's expected format exactly
2. **Filter outliers**: Remove conversations that are too long (> 20 turns) or too short (< 2 turns)
3. **Reduce learning rate**: Start with a lower learning rate (e.g., 1e-5 instead of 1e-4)
4. **Add regularization**: Use gradient clipping and weight decay

:::


---

## What You've Learned

You now have hands-on experience with:

- ✓ Understanding SFT and DPO training data formats
- ✓ Filtering rollouts for quality before processing
- ✓ Converting rollouts into training-ready formats
- ✓ Validating and evaluating training data quality

**Key insight**: High-quality training data comes from careful filtering and processing of rollouts. Start with good data, and your models will learn better behaviors.

---

## Next Steps

You've completed offline training data preparation! Continue with:

- **[Configuration Management](09-configuration-guide.md)**: Master NeMo Gym's flexible configuration system
- **[Rollout Collection Fundamentals](../about/concepts/rollout-collection-fundamentals.md)**: Deep dive into advanced collection strategies

Or explore the [Concepts](../about/concepts/index.md) section for deeper understanding of the framework.
