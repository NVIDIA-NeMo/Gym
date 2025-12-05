(tutorial-train-with-trl)=

# Train with TRL (Offline)

In {doc}`Collecting Rollouts </get-started/rollout-collection>`, you generated scored interactions between your agent and environment. Now you'll filter those rollouts, format them for TRL, and run SFT training.

:::{card}

**Goal**: Fine-tune a model on your Gym rollouts using TRL's SFTTrainer.

^^^

**In this tutorial, you will**:

1. Filter rollouts by reward score
2. Convert to TRL's expected format
3. Configure and run SFT training
4. Validate the trained model with Gym

:::

:::{button-ref} /tutorials/integrate-training-frameworks/index
:color: secondary
:outline:
:ref-type: doc

← Training Frameworks
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`Collecting Rollouts </get-started/rollout-collection>`
- ✅ Rollouts file at `results/rollouts.jsonl`
- ✅ TRL installed (`pip install trl transformers datasets`)
- ✅ GPU with sufficient memory for your base model

**What you'll build**: A fine-tuned model checkpoint trained on successful agent behaviors from your rollouts.

---

## 1. Filter High-Quality Rollouts

For SFT, you want only successful rollouts. Filter by reward score:

```python
import json

def filter_rollouts(input_file: str, output_file: str, min_reward: float = 0.8):
    """Keep only rollouts above the reward threshold."""
    kept, total = 0, 0
    
    with open(input_file) as f, open(output_file, 'w') as out:
        for line in f:
            rollout = json.loads(line)
            total += 1
            
            if rollout.get('reward', 0) >= min_reward:
                out.write(line)
                kept += 1
    
    print(f"Kept {kept}/{total} rollouts ({kept/total*100:.1f}%)")
    return kept

# Filter rollouts with reward >= 0.8
filter_rollouts('results/rollouts.jsonl', 'filtered_rollouts.jsonl', min_reward=0.8)
```

```{list-table} Filter Parameters
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Default
  - Description
* - `min_reward`
  - `0.8`
  - Minimum verification score to include
* - `min_turns`
  - None
  - Minimum conversation turns (optional)
* - `max_turns`
  - None
  - Maximum conversation turns (optional)
```

**✅ Success Check**: You should have at least 100+ filtered rollouts for meaningful training.

---

## 2. Convert to TRL Format

TRL's SFTTrainer expects a `messages` field. Extract from your rollouts:

```python
import json
from datasets import Dataset

def rollouts_to_sft_dataset(rollout_file: str) -> Dataset:
    """Convert filtered rollouts to HuggingFace Dataset for TRL."""
    examples = []
    
    with open(rollout_file) as f:
        for line in f:
            rollout = json.loads(line)
            examples.append({
                "messages": rollout["output"]  # The conversation from the rollout
            })
    
    return Dataset.from_list(examples)

# Create dataset
dataset = rollouts_to_sft_dataset('filtered_rollouts.jsonl')
print(f"Dataset size: {len(dataset)}")
print(f"Example: {dataset[0]}")
```

**✅ Success Check**: Each example has a `messages` field with the conversation.

---

## 3. Configure TRL Training

Set up the SFTTrainer with your model and dataset:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Load base model (use the same model you collected rollouts with)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Or your model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure training
config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
    max_seq_length=2048,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

```{list-table} Key Training Parameters
:header-rows: 1
:widths: 30 70

* - Parameter
  - Recommendation
* - `num_train_epochs`
  - 1–3 for fine-tuning; more risks overfitting
* - `learning_rate`
  - 1e-5 to 5e-5 for fine-tuning
* - `per_device_train_batch_size`
  - As large as GPU memory allows
* - `max_seq_length`
  - Match your rollout collection settings
```

---

## 4. Run Training

Start the training loop:

```python
# Train
trainer.train()

# Save the final model
trainer.save_model("./sft_final")
tokenizer.save_pretrained("./sft_final")
```

**✅ Success Check**: Training loss should decrease over time. Final loss typically 0.5–2.0 depending on task.

---

## 5. Validate with Gym

Test your trained model by collecting new rollouts:

```bash
# Point Gym at your trained model
# Update your model config to use ./sft_final

ng_collect_rollouts +agent_name=your_agent \
    +input_jsonl_fpath=evaluation_tasks.jsonl \
    +output_jsonl_fpath=post_training_rollouts.jsonl \
    +limit=50
```

Compare metrics:

```python
import json

def compute_metrics(rollout_file: str):
    rewards = []
    with open(rollout_file) as f:
        for line in f:
            rollout = json.loads(line)
            rewards.append(rollout.get('reward', 0))
    
    return {
        'avg_reward': sum(rewards) / len(rewards),
        'success_rate': sum(1 for r in rewards if r >= 0.5) / len(rewards)
    }

baseline = compute_metrics('results/rollouts.jsonl')
trained = compute_metrics('post_training_rollouts.jsonl')

print(f"Baseline: {baseline}")
print(f"Trained:  {trained}")
```

**✅ Success Check**: Average reward and success rate should improve over baseline.

---

## DPO Training (Alternative)

If you have paired rollouts (2 per prompt with different rewards), use DPO instead:

```python
def create_dpo_pairs(rollout_file: str, output_file: str, min_diff: float = 0.1):
    """Create preference pairs from rollouts."""
    # Group by prompt
    task_groups = {}
    with open(rollout_file) as f:
        for line in f:
            rollout = json.loads(line)
            task_id = hash(json.dumps(rollout['responses_create_params']['input']))
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(rollout)
    
    # Create pairs
    pairs = []
    for rollouts in task_groups.values():
        if len(rollouts) >= 2:
            rollouts.sort(key=lambda x: x['reward'], reverse=True)
            if rollouts[0]['reward'] - rollouts[1]['reward'] >= min_diff:
                pairs.append({
                    "prompt": rollouts[0]['responses_create_params']['input'],
                    "chosen": rollouts[0]['output'],
                    "rejected": rollouts[1]['output']
                })
    
    with open(output_file, 'w') as out:
        for pair in pairs:
            out.write(json.dumps(pair) + '\n')
    
    print(f"Created {len(pairs)} preference pairs")

# Then use TRL's DPOTrainer instead of SFTTrainer
```

Refer to {doc}`/training/datasets/format-specification` for format details.

---

## Troubleshooting

:::{dropdown} Out of memory during training
Reduce `per_device_train_batch_size` or `max_seq_length`. Enable gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```
:::

:::{dropdown} Training loss not decreasing
- Check data format matches model's chat template
- Reduce learning rate
- Verify filtered rollouts are actually high quality
:::

:::{dropdown} Model performance worse after training
- You may be overfitting — reduce epochs or dataset size
- Filter more aggressively (higher `min_reward`)
- Check for data quality issues in rollouts
:::

---

## Learn More

For deeper understanding of the concepts used in this tutorial:

- {doc}`/training/datasets/format-specification` — SFT and DPO format schemas
- {doc}`/training/rollout-collection/configure-sampling` — How rollout sampling affects training data quality
- {doc}`/training/verification/index` — Verification patterns that determine reward scores

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Iterate
:link: /get-started/rollout-collection
:link-type: doc

Collect more rollouts with your improved model.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Scale Up
:link: train-with-nemo-rl
:link-type: doc

For on-policy RL training, try NeMo RL.
:::

::::
