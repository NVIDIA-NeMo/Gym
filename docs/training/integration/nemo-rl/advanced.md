(training-integration-nemo-rl-advanced)=

# Advanced Integration

Advanced patterns, optimization techniques, and production workflows for NeMo Gym + NeMo RL.

```{contents}
:local:
:depth: 2
```

---

### Multi-Turn Conversations

For multi-turn tool use or dialogue:

```python
def transform_multiturn_rollout(rollout):
    """Transform multi-turn conversation to OpenAI format."""
    messages = []
    
    # Include full conversation history
    for turn in rollout["output"]:
        if turn["type"] == "message" and turn.get("role") == "user":
            messages.append({
                "role": "user",
                "content": turn["content"]
            })
        elif turn["type"] == "message" and turn.get("role") == "assistant":
            messages.append({
                "role": "assistant",
                "content": turn["content"]
            })
        elif turn["type"] == "function_call":
            # Handle tool calls in multi-turn
            # ...
    
    return {"messages": messages}
```

Configure NeMo RL for multi-turn:

```yaml
grpo:
  max_rollout_turns: 3  # Allow up to 3 turns of interaction
```

### Custom Reward Shaping

Map Gym's reward distribution to NeMo RL's expected range:

```yaml
grpo:
  reward_scaling:
    enabled: true
    source_min: 0.0      # Your Gym reward range
    source_max: 1.0
    target_min: -1.0     # Scale to [-1, 1] for training
    target_max: 1.0
```

### Combining Multiple Resource Servers

If you collected rollouts from multiple Gym resource servers:

```python
# Transform and combine
for server_name in ["math", "code", "instruction_following"]:
    transform_rollouts(
        f"rollouts_{server_name}.jsonl",
        f"transformed_{server_name}.jsonl"
    )

# Concatenate for multi-task training
!cat transformed_*.jsonl > combined_train.jsonl
```

Configure multi-task:

```yaml
data:
  dataset_name: ResponseDataset
  train_data_path: "combined_train.jsonl"
  # Each rollout should have a "task_name" field
```

---

## Performance Optimization

### Data Collection Strategy

For efficient NeMo RL training:

1. **GRPO**: Collect diverse prompts (1 rollout each) rather than multiple rollouts per prompt
   ```yaml
   # In NeMo Gym collection
   num_repeats: 1
   num_samples_in_parallel: 50  # Diverse prompts
   ```

2. **SFT**: Collect with low temperature for consistent high-quality examples
   ```yaml
   responses_create_params:
     temperature: 0.2
   ```

3. **DPO**: Collect with higher temperature for quality variation
   ```yaml
   responses_create_params:
     temperature: 0.7
   num_repeats: 2  # Multiple per prompt for pairing
   ```

### Training Throughput

Maximize NeMo RL training speed:

1. **Enable sequence packing** (already default):
   ```yaml
   policy:
     sequence_packing:
       enabled: true
   ```

2. **Use colocated generation** for GRPO (shares GPUs):
   ```yaml
   policy:
     generation:
       colocated:
         enabled: true
   ```

3. **Optimize batch sizes** for your GPU memory:
   ```yaml
   policy:
     train_micro_batch_size: 4      # Adjust for GPU size
     train_global_batch_size: 512   # Keep high for stability
   ```

---

## Migration Checklist

Use this checklist when integrating NeMo Gym data with NeMo RL:

**Data Preparation**
- [ ] Collected rollouts with verification ({doc}`../rollout-collection/index`)
- [ ] Validated reward distribution ({doc}`../data-quality/index`)
- [ ] Transformed data to NeMo RL format (GRPO/SFT/DPO)
- [ ] Created train/validation splits
- [ ] Verified transformed data structure

**NeMo RL Setup**
- [ ] Cloned NeMo RL and set up environment
- [ ] Set environment variables (`HF_HOME`, `WANDB_API_KEY`)
- [ ] Logged in to HuggingFace (`huggingface-cli login`)
- [ ] Created training configuration YAML
- [ ] Specified correct dataset paths and keys

**Training Launch**
- [ ] Ran test run on 1 GPU to verify data loading
- [ ] Configured checkpointing and logging
- [ ] Set appropriate batch sizes for available GPUs
- [ ] Launched full training run
- [ ] Monitoring metrics (loss, reward, accuracy)

**Validation**
- [ ] Checked training loss decreases
- [ ] Reviewed generated samples
- [ ] Evaluated on held-out validation set
- [ ] Compared to baseline model

---

## Next Steps

### After Training

1. **Evaluate trained model**: Use NeMo RL's evaluation tools
   ```bash
   uv run python examples/run_eval.py \
       generation.model_name=$PWD/results/grpo/hf
   ```

2. **Iterate on data**: Collect more rollouts targeting weaknesses
3. **Try different algorithms**: Compare GRPO, SFT, DPO on your task
4. **Scale up**: Move to larger models or multi-node training

### Resources

**NeMo RL Documentation**:
- [Training Backends](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/design-docs/training-backends.md)
- [GRPO Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md)
- [SFT Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md)
- [DPO Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/dpo.md)

**NeMo Gym Documentation**:
- {doc}`../rollout-collection/optimize-for-training/index` - Optimize data collection
- {doc}`../verification/custom-patterns-cookbook` - Advanced verification patterns
- {doc}`../data-quality/index` - Filter and curate data

---

## Related Topics

- {doc}`../handoff-to-training` - Overview of training framework handoff
- {doc}`../datasets/prepare-for-training` - Algorithm-specific data preparation
- {doc}`../rollout-collection/sampling-strategies/index` - Collection strategies by algorithm
- {doc}`../verification/index` - Verification and reward design

