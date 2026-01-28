(on-policy-training)=

# On-Policy Training

## What is On-Policy Training?

On-policy training refers to the scenario where the policy (or model) we are training matches the policy used to generate rollouts. This requires that the token ids and log probabilities (logprobs) are the same in the training and inference parts of our training algorithm. There are various scenarios that lead to subtle off policy training. Off policy training can be unstable and collapse, but sometimes it is fine or desired.

The mismatch between the training and inference token ids and logprobs often leads to large gradients that lead to training collapse. This can occur due to differences in the training and inference algorihms and kernels, such as vLLM vs Megatron-core. Additionally, retokenization of generated tokens can lead to mismatches (see {doc}`openai-compatible-http-server-on-policy-correction`). Multi-step and multi-turn agent environments also lead to mismatch if the trajectory is handled in any way other than exactly what the model sees during the rollout. For example, with Qwen3 thinking models, reasoning traces in previous turns are not stored in later turns during inference. Therefore, when we recompute the logprobs with the training policy, we are recomputing without these previous thinking blocks, leading to a mismatch in logprobs of generated tokens between inference and train policy. More complex agents, that do context management, or multi-agent systems, lead to even further complexity.

## Configuration

By default, NeMo-Gym and NeMo-RL enforce monotonicity or strictly increasing trajectories in multi-turn scenarios. This means we do not allow dropping previous thinking blocks or other forms of context management.

```{warning}
Disabling on-policy enforcement may lead to training instability. Use only when necessary and monitor training metrics closely.
```

To disable enforcement, use:

```yaml
# Disable monotonicity enforcement and on-policy token ID correction
enforce_monotonicity: false  # TODO: Implement this. Not supported yet. RL issue #1812
```

## Recommended Approaches

**For Qwen3 thinking mode**:
1. **Preferred**: Train with reasoning truncation disabled (keep reasoning across all turns)
2. **Alternative**: Train models with no reasoning component

**For complex agents with context management**:
- Consider whether history modification is necessary
- If required, disable enforcement and monitor training stability
- Use techniques like importance sampling to handle off-policy data

## Related Topics

- {doc}`openai-compatible-http-server-on-policy-correction` - Detailed technical explanation and token ID fix
- {doc}`generation-backend-and-openai-compatible-http-server` - Generation backend requirements
