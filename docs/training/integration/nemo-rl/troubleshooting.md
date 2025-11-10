(training-integration-nemo-rl-troubleshooting)=

# Troubleshooting

Common issues and solutions when integrating NeMo Gym with NeMo RL.

```{contents}
:local:
:depth: 2
```

---

### Data Issues

**Problem**: `KeyError: 'input'` or `KeyError: 'output'`

**Solution**: Check your transformation script is correctly mapping fields. Verify with:

```bash
head -n 1 transformed_data.jsonl | python -m json.tool
```

---

**Problem**: "No pairs created" for DPO

**Solution**: 
- Check reward distribution: `jq '.reward' rollouts.jsonl | sort | uniq -c`
- Lower `--min_gap` threshold
- Adjust `--high_threshold` and `--low_threshold`
- Ensure you have rollouts with varied quality

---

**Problem**: SFT dataset too small (< 100 examples)

**Solution**:
- Lower `--reward_threshold` (e.g., from 0.95 to 0.80)
- Collect more rollouts with diverse sampling strategies
- Check if your verification is too strict ({doc}`../verification/validate-verification`)

---

### Training Issues

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Reduce batch sizes:
   ```yaml
   policy:
     train_micro_batch_size: 2  # Reduce from 4
     train_global_batch_size: 256  # Reduce proportionally
   ```

2. Enable gradient checkpointing:
   ```yaml
   policy:
     dtensor_cfg:
       activation_checkpointing: true
   ```

3. Use smaller model or tensor parallelism:
   ```yaml
   policy:
     model_name: "Qwen/Qwen2.5-1.5B"  # Instead of 3B
     dtensor_cfg:
       tensor_parallel_size: 2  # Split across 2 GPUs
   ```

---

**Problem**: Training loss not decreasing

**Solution**:
1. Check data quality:
   ```python
   # Verify rewards make sense
   import json
   rollouts = [json.loads(line) for line in open('rollouts.jsonl')]
   print(f"Mean reward: {sum(r['reward'] for r in rollouts)/len(rollouts):.3f}")
   ```

2. Adjust learning rate:
   ```yaml
   policy:
     optimizer:
       kwargs:
         lr: 1.0e-5  # Lower if loss spikes
   ```

3. Check for data format issues:
   - Ensure transformed data has correct fields
   - Validate messages are properly formatted
   - Check for empty or malformed outputs

---

**Problem**: vLLM generation errors in GRPO

**Solution**:
1. Check vLLM configuration matches model:
   ```yaml
   policy:
     generation:
       vllm_cfg:
         max_model_len: 1024  # Match max_total_sequence_length
         gpu_memory_utilization: 0.6
   ```

2. Enable eager mode for debugging:
   ```yaml
   policy:
     generation:
       vllm_cfg:
         enforce_eager: true  # Slower but more stable
   ```

---

### Environment/Verification Issues

**Problem**: GRPO rewards all 0.0

**Solution**: GRPO uses its own verification environment during training, not your Gym rewards. Ensure:

1. Ground truth is included in transformed data:
   ```python
   # Check transformed data has ground_truth field
   import json
   data = json.loads(open('grpo_data/train.jsonl').readline())
   print("Has ground_truth:", "ground_truth" in data)
   ```

2. Environment is configured:
   ```yaml
   env:
     math:
       math_verify_impl: "hf_math_verify"
   ```

3. Prompts include necessary context for environment to verify

---

