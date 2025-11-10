(training-integration-nemo-rl-advanced)=

# Advanced Integration

Production workflows, optimization strategies, and advanced patterns for NeMo Gym + NeMo RL integration.

:::{admonition} Prerequisites
:class: tip
This guide assumes you're familiar with the basic integration workflow described in {doc}`index`. You should have successfully completed at least one training run before tackling these advanced topics.
:::

---

## Multi-Task Training

### Combining Multiple Resource Servers

Train a single model on rollouts from multiple NeMo Gym resource servers:

::::{tab-set}

:::{tab-item} Workflow

**Step 1: Collect from Each Server**

```bash
# Math tasks
ng_collect_rollouts \
  +agent_name=math_agent \
  +output_jsonl_fpath=rollouts_math.jsonl

# Code tasks  
ng_collect_rollouts \
  +agent_name=code_agent \
  +output_jsonl_fpath=rollouts_code.jsonl
```

**Step 2: Transform and Combine**

```bash
# Transform each dataset
python transform.py rollouts_math.jsonl > math_train.jsonl
python transform.py rollouts_code.jsonl > code_train.jsonl

# Combine
cat math_train.jsonl code_train.jsonl > combined_train.jsonl
```

**Step 3: Configure NeMo RL**

```yaml
data:
  train_data_path: "combined_train.jsonl"
  dataset_name: ResponseDataset
```

:::

:::{tab-item} Best Practices

**Balance dataset sizes**:
- Ensure each task has sufficient examples (at least 100-500)
- Consider upsampling smaller datasets to prevent imbalance
- Monitor per-task performance during training

**Task identification**:
- Add `task_name` field to each rollout during transformation
- Use task-specific metrics for evaluation
- Consider task-specific reward scaling if needed

**Quality filtering**:
- Apply consistent quality thresholds across all tasks
- Remove low-reward rollouts (< 0.5) unless specifically needed
- Verify data balance after filtering

:::

::::

---

## Reward Shaping

### Understanding Reward Distributions

NeMo Gym resource servers return rewards in different ranges. Check your data before training:

```python
import json
from collections import Counter

# Load rollouts
rollouts = [json.loads(line) for line in open("rollouts.jsonl")]
rewards = [r["reward"] for r in rollouts]

print(f"Reward range: {min(rewards):.2f} to {max(rewards):.2f}")
print(f"Mean: {sum(rewards)/len(rewards):.2f}")
print(f"Distribution: {Counter([round(r, 1) for r in rewards])}")
```

::::{dropdown} **When to Apply Reward Scaling**

**Apply scaling when**:
- Rewards are outside [-1, 1] range
- Most rewards cluster at extremes (0.0 or 1.0)
- Combining datasets with different reward ranges

**Skip scaling when**:
- Rewards already in [-1, 1]
- Training SFT (only uses high-reward data)
- Reward distribution is well-balanced

::::

### Scaling Configuration

Refer to [NeMo RL's reward normalization documentation](https://github.com/NVIDIA-NeMo/RL) for configuration options. Most NeMo RL algorithms handle reward normalization automatically.

---

## Performance Optimization

### Data Collection Strategies

Optimize rollout collection based on your training algorithm:

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Algorithm
  - Collection Strategy
  - Configuration
* - **GRPO**
  - Diverse prompts, single rollout each
  - High parallelism, moderate temperature
* - **SFT**
  - High-quality demos, low temperature
  - Filter for reward ≥ 0.8
* - **DPO**
  - Multiple rollouts per prompt
  - Higher temperature for variation
```

::::{dropdown} **Detailed Collection Parameters**

**For GRPO Training**:

```yaml
# Prioritize prompt diversity
num_samples_in_parallel: 50  # Process many different prompts
responses_create_params:
  temperature: 0.5  # Moderate exploration
```

**For SFT Training**:

```yaml
# Prioritize quality over diversity
responses_create_params:
  temperature: 0.2  # Low temperature for consistency
# Then filter for reward >= 0.8 during transformation
```

**For DPO Training**:

```yaml
# Collect multiple rollouts per prompt
num_repeats: 2  # or more for better pairing
responses_create_params:
  temperature: 0.7  # Higher for quality variation
```

Refer to {doc}`../../rollout-collection/optimize-for-training/index` for comprehensive collection optimization.

::::

### Training Configuration

NeMo RL provides extensive training configuration options. Key areas to optimize:

**Batch Size Tuning**:
- Start with recommended defaults from NeMo RL examples
- Increase `global_batch_size` for stable training
- Adjust `micro_batch_size` based on GPU memory

**Memory Management**:
- Use gradient checkpointing for large models
- Enable mixed precision training (FP16/BF16)
- Consider model parallelism for 30B+ models

**Generation Settings** (for GRPO):
- Use colocated generation to share GPUs with training
- Tune generation parameters (temperature, top-p) for diversity

Refer to [NeMo RL's training guides](https://github.com/NVIDIA-NeMo/RL/tree/main/docs/guides) for algorithm-specific optimization.

---

## Production Deployment Checklist

Use this checklist when deploying NeMo Gym → NeMo RL integration to production:

::::{tab-set}

:::{tab-item} Data Workflow

**Rollout Collection**
- [ ] Resource server verified and tested ({doc}`../../verification/index`)
- [ ] Collection scripts automated and scheduled
- [ ] Quality metrics monitored (avg reward, completion rate)
- [ ] Data versioning implemented (track which rollouts → which model)

**Data Transformation**
- [ ] Transformation scripts tested on sample data
- [ ] Output format validated against NeMo RL requirements
- [ ] Train/validation splits created (typically 90/10 or 95/5)
- [ ] Data quality checks automated (reward distribution, field validation)

**Data Management**
- [ ] Storage solution configured (S3, GCS, or local)
- [ ] Backup and retention policies defined
- [ ] Access controls implemented
- [ ] Dataset documentation maintained

:::

:::{tab-item} Training Setup

**Environment**
- [ ] NeMo RL repository cloned and tested
- [ ] Environment variables configured (`HF_HOME`, `WANDB_API_KEY`)
- [ ] GPU resources allocated and tested
- [ ] HuggingFace authentication configured (if needed)

**Configuration**
- [ ] Training config YAML created and validated
- [ ] Hyperparameters tuned on small test set
- [ ] Checkpointing and logging configured
- [ ] Experiment tracking set up (W&B, TensorBoard)

**Validation**
- [ ] Test run completed on 1 GPU
- [ ] Data loading verified (no errors, correct batch sizes)
- [ ] Training metrics logged successfully
- [ ] Resource usage monitored (GPU memory, disk space)

:::

:::{tab-item} Monitoring & Iteration

**During Training**
- [ ] Training loss decreasing
- [ ] Generated samples reviewed for quality
- [ ] Resource utilization within limits
- [ ] No OOM or other runtime errors

**After Training**
- [ ] Model evaluated on held-out validation set
- [ ] Performance compared to baseline
- [ ] Model artifacts saved and versioned
- [ ] Results documented and shared

**Continuous Improvement**
- [ ] Identify failure modes from evaluation
- [ ] Collect targeted rollouts for weaknesses
- [ ] Experiment with different algorithms
- [ ] Scale to larger models or datasets

:::

::::

---

## Related Documentation

### NeMo RL Resources

- [NeMo RL Repository](https://github.com/NVIDIA-NeMo/RL) - Main codebase and examples
- [Training Guides](https://github.com/NVIDIA-NeMo/RL/tree/main/docs/guides) - GRPO, SFT, DPO guides
- [Design Docs](https://github.com/NVIDIA-NeMo/RL/tree/main/docs/design-docs) - Architecture and training backends

### NeMo Gym Guides

- {doc}`../../rollout-collection/optimize-for-training/index` - Optimize data collection
- {doc}`../../verification/custom-patterns-cookbook` - Advanced verification patterns
- {doc}`../../data-quality/index` - Filter and curate rollouts
- {doc}`../../datasets/prepare-for-training` - Algorithm-specific data preparation
