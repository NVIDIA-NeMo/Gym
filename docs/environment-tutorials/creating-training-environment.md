(env-creating-training-environment)=
# Creating a Training Environment

Build a complete training environment ready for reinforcement learning with NeMo RL.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
1-2 hours
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`/get-started/detailed-setup`
- Completed {doc}`/tutorials/creating-resource-server`

:::

::::

---

## Overview

You've built a resources server with tools and verification. Now learn how to transform it into a **training environment** — one that can effectively train models via reinforcement learning.

A training environment differs from a simple resources server in three key ways:

| Aspect | Resources Server | Training Environment |
|--------|------------------|----------------------|
| **Data scale** | 5-10 example inputs | Thousands of training examples |
| **Reward design** | Basic pass/fail | Carefully shaped for learning |
| **Evaluation** | Manual testing | Systematic baseline and metrics |

---

## Task Design Principles

Good RL tasks share common characteristics that make them learnable.

### Define Clear Success Criteria

Before implementing, answer these questions:

1. **What does success look like?** Define the exact conditions for reward=1.0
2. **What partial progress exists?** Identify intermediate states worth rewarding
3. **What are failure modes?** Understand how the model might "cheat" or fail

```python
# Example: Clear success criteria for a search task
SUCCESS_CRITERIA = {
    "found_answer": True,           # Primary goal
    "used_correct_tool": True,      # Required behavior
    "max_steps": 5,                 # Efficiency constraint
}
```

### Choose Appropriate Complexity

Start simple and increase complexity gradually:

| Complexity | Example | Training Consideration |
|------------|---------|------------------------|
| **Single-step** | Math calculation | Fast iteration, clear signal |
| **Multi-step** | Information retrieval | Requires intermediate rewards |
| **Multi-turn** | Dialogue | Needs conversation-level rewards |

:::{tip}
Begin with single-step tasks. Add complexity only after achieving good performance on simpler variants.
:::

---

## Designing Effective Rewards

Reward design is the most critical aspect of training environment quality.

### Reward Signal Properties

**Density**: How often does the model receive feedback?

```python
# Sparse reward — only at task completion
reward = 1.0 if task_complete else 0.0

# Dense reward — feedback at each step
reward = step_progress + correctness_bonus
```

**Scale**: Keep rewards in a consistent range (typically 0.0 to 1.0).

**Informativeness**: Rewards should distinguish good actions from bad ones.

### Reward Patterns

#### Binary Rewards

Use for tasks with clear right/wrong answers:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    correct = extract_answer(body.response) == body.expected_answer
    reward = 1.0 if correct else 0.0
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Math, factual Q&A, code correctness

#### Partial Credit

Use when intermediate progress is meaningful:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    steps = extract_steps(body.response)
    correct_steps = sum(1 for s in steps if is_correct(s))
    reward = correct_steps / len(steps)
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Multi-step reasoning, structured outputs

#### Shaped Rewards

Combine multiple signals for complex tasks:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    # Primary objective
    task_score = 1.0 if task_complete(body.response) else 0.0
    
    # Secondary objectives
    efficiency_bonus = max(0, 1 - (num_steps / max_steps)) * 0.2
    format_bonus = 0.1 if correct_format(body.response) else 0.0
    
    reward = task_score + efficiency_bonus + format_bonus
    reward = min(reward, 1.0)  # Cap at 1.0
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Complex workflows, multi-objective tasks

### Avoiding Common Pitfalls

:::{warning}
**Reward hacking**: Models may find unintended shortcuts. Test edge cases.

```python
# Bad: Rewards length, not quality
reward = len(response) / 1000  # Model learns to be verbose

# Good: Rewards actual task completion
reward = 1.0 if answer_matches_expected else 0.0
```
:::

---

## Building Your Training Dataset

Training requires significantly more data than testing.

### Data Requirements

| Dataset | Size | Purpose |
|---------|------|---------|
| **Training** | 1,000 - 100,000+ | Model learns from these |
| **Validation** | 100 - 1,000 | Monitor training progress |
| **Test** | 100 - 500 | Final evaluation (held out) |

### Creating Training Data

#### From Existing Datasets

```python
import json
from datasets import load_dataset

# Load a HuggingFace dataset
dataset = load_dataset("gsm8k", "main", split="train")

# Convert to NeMo Gym format
with open("data/train.jsonl", "w") as f:
    for example in dataset:
        item = {
            "responses_create_params": {
                "input": [{"role": "user", "content": example["question"]}]
            },
            "expected_answer": example["answer"].split("####")[-1].strip()
        }
        f.write(json.dumps(item) + "\n")
```

#### Synthetic Data Generation

For custom tasks, generate examples programmatically:

```python
import random
import json

def generate_math_example():
    a, b = random.randint(1, 100), random.randint(1, 100)
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": f"Calculate {a} * {b}"}]
        },
        "expected_answer": str(a * b)
    }

with open("data/train.jsonl", "w") as f:
    for _ in range(10000):
        f.write(json.dumps(generate_math_example()) + "\n")
```

### Data Quality Checklist

Before training, verify your data:

- [ ] **Balanced difficulty**: Mix of easy, medium, hard examples
- [ ] **Diverse inputs**: Varied phrasing, edge cases included
- [ ] **Correct labels**: Expected answers are accurate
- [ ] **No data leakage**: Test set doesn't overlap with training

---

## Evaluation and Iteration

### Establish Baselines

Before training, measure baseline performance:

```bash
# Collect rollouts with your current model
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=data/validation.jsonl \
    +output_jsonl_fpath=data/baseline_rollouts.jsonl
```

Analyze the results:

```python
import json

rewards = []
with open("data/baseline_rollouts.jsonl") as f:
    for line in f:
        data = json.loads(line)
        rewards.append(data.get("reward", 0))

print(f"Baseline accuracy: {sum(r == 1.0 for r in rewards) / len(rewards):.2%}")
print(f"Average reward: {sum(rewards) / len(rewards):.3f}")
```

### Monitor Training Progress

Track these metrics during training:

| Metric | What It Tells You |
|--------|-------------------|
| **Mean reward** | Overall performance trend |
| **Reward variance** | Learning stability |
| **Success rate** | Percentage of fully correct responses |
| **Step count** | Efficiency of solutions |

### Common Failure Modes

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Reward stays flat | Reward too sparse | Add intermediate rewards |
| Reward oscillates | Learning rate too high | Reduce LR, increase batch size |
| Model "cheats" | Reward hacking | Tighten verification logic |
| Performance plateaus | Task too hard | Simplify or add curriculum |

---

## Connecting to Training

Once your environment is ready, connect it to NeMo RL for training.

### Prepare Configuration

Create a training configuration that references your environment:

```yaml
# configs/my_training.yaml
training:
  algorithm: grpo
  environment:
    resources_server: my_resources_server
    agent: simple_agent
  data:
    train_jsonl: data/train.jsonl
    validation_jsonl: data/validation.jsonl
```

### Run Training

```bash
# See NeMo RL documentation for full training commands
# This is a simplified example
python -m nemo_rl.train \
    --config configs/my_training.yaml \
    --output_dir results/my_experiment
```

:::{seealso}
For complete training instructions, see {ref}`training-nemo-rl-grpo-index`.
:::

---

## Next Steps

After creating your training environment:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-a-Judge
:link: llm-as-judge
:link-type: doc
Use LLMs for flexible verification of open-ended tasks.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Environments
:link: multi-step
:link-type: doc
Build sequential tool-calling workflows.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training models on your environment.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Prepare Data
:link: /data/prepare-validate
:link-type: doc
Learn more about data formats and validation.
:::

::::

---

## Summary

You've learned how to:

✅ Design tasks with clear success criteria  
✅ Create effective reward functions for RL  
✅ Build training datasets at scale  
✅ Establish baselines and monitor progress  
✅ Connect your environment to training  

Your training environment is now ready for reinforcement learning with NeMo RL!
