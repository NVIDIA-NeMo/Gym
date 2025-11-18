# Offline Training with Rollouts (SFT/DPO)

**Goal**: Transform your generated rollouts into high-quality training data for supervised fine-tuning (SFT) and direct preference optimization (DPO).

## Why Offline Training?

**Offline training** uses pre-collected rollouts to improve AI models without real-time exploration. This approach is ideal when:

- You have a working agent that demonstrates good behaviors
- You want reproducible results - same data, consistent training outcomes
- You need cost-effective training - no expensive exploration during training
- You want to capture expert demonstrations - preserve successful patterns
- You have limited compute - more efficient than reinforcement learning

**The offline training pipeline**: Generate rollouts → Filter and process → Train models → Deploy improved agents

## Training Data Types

### Supervised Fine-Tuning (SFT) Data

**Purpose**: Train models to follow successful agent interaction patterns

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

### Direct Preference Optimization (DPO) Data

**Purpose**: Train models to prefer better responses over worse ones

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

## Data Preparation Overview

The offline training pipeline follows this logical flow:

1. Collect rollouts using strategies from [Tutorial 5]
- **SFT data**: Use consistent generation (low temperature, single rollout per task)
- **DPO data**: Use diverse generation (higher temperature, 2 rollouts per task for comparison)
1. Filter for quality - Remove poor rollouts before processing
2. Format for training - Convert to SFT or DPO format based on your goals



## Step 1: Quality Filtering and Curation

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

**Note**: These are example filtering approaches. Customize the criteria, thresholds, and sampling strategies based on your specific domain and quality requirements.

## Step 2: Format for Training

Once you have filtered, high-quality rollouts, format them for your chosen training method:

### SFT Data Processing

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

### DPO Data Processing

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

## Best Practices

### 1. Data Quality Over Quantity

```python
# Prefer high-quality filtered data over large noisy datasets
filter_criteria = {
    'min_reward': 0.8,        # High threshold for SFT
    'min_success_rate': 0.9,
    'require_tool_usage': True  # Domain-specific requirements
}
```

### 2. Balanced Datasets

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

### 3. Iterative Improvement

```bash
# Iteration cycle
1. Generate rollouts with current agent
2. Filter and prepare training data  
3. Train improved model
4. Deploy and evaluate
5. Use improved agent to generate better rollouts
```

### 4. Version Control

```bash
# Track data versions
mkdir -p training/v1.0/
mv sft_data.jsonl training/v1.0/
mv dpo_pairs.jsonl training/v1.0/

# Track model versions  
mkdir -p models/agent_v1.0/
cp -r ./results/* models/agent_v1.0/
```

## Troubleshooting

### Problem: Poor Training Data Quality

```
Low average rewards, inconsistent behaviors
```

**Solutions**:
- Increase `min_reward` threshold for filtering
- Generate rollouts with lower temperature (more consistent)
- Add manual curation step
- Improve base agent before data collection

### Problem: Insufficient Data Diversity

```
Model overfits to limited patterns
```

**Solutions**:
- Generate rollouts with higher temperature
- Use more diverse input tasks
- Collect data from multiple agent configurations
- Balance dataset across task types

### Problem: Training Instability

```
Loss doesn't converge, model performance degrades
```

**Solutions**:
- Check data format compatibility with training framework
- Reduce learning rate
- Add regularization
- Filter out extremely long or short conversations

## What You've Learned

You now know how to transform rollouts into training data:

- **Data preparation strategies** for SFT and DPO
- **Quality filtering and curation** techniques  
- **Evaluation methods** to measure improvement
- **Best practices** for sustainable offline training workflows
