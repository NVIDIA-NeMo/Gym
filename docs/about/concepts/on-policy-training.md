(on-policy-training)=

# On-Policy Training

In reinforcement learning, **on-policy training** means the model you're training is the same model that generated the rollouts. When token IDs and log probabilities (logprobs) match between generation and training, you have on-policy training. When they don't match, you have off-policy training—which can cause instability or collapse, though in some cases it may be acceptable or even desired.

## Why On-Policy Matters

Policy optimization algorithms backpropagate through a loss calculated using logprobs. When the logprobs computed during training differ from those computed during generation, the gradients become unreliable. Small mismatches are tolerable, but large mismatches typically cause training runs to crash.

### Common Causes of Mismatch

Several scenarios lead to train-generation mismatch, including differences in training and inference algorithms or kernels (such as vLLM vs Megatron-core):

**Re-tokenization**
: When generated tokens are de-tokenized to strings and then re-tokenized for the next model call, the token IDs can change. For example, tokens that de-tokenize to `"_Ski" + "nny"` might re-tokenize as a single `"_Skinny"` token.

**Re-chat templating**
: When the model's output is parsed into structured objects (like tool calls) and then re-rendered through a chat template, the formatting can differ from the original generation.

**Non-monotonic history**
: When rollout history is modified during execution—such as truncating reasoning traces or summarizing context—the prompt token IDs at training time differ from those seen during generation.

:::{tip}
For a detailed technical explanation of these problems and their solutions, refer to {doc}`../../contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction`.
:::

### Example: Reasoning Models

Models like Qwen3 that produce reasoning traces present a specific challenge. Some chat templates remove reasoning from previous turns during inference, but training sees the full trajectory. This creates a logprob mismatch because the model computes probabilities against different context.

## Configuration

By default, NeMo-Gym and NeMo-RL enforce monotonicity (strictly increasing trajectories) in multi-turn scenarios. This prevents dropping previous thinking blocks or other forms of context management that would cause train-generation mismatch.

:::{warning}
Disabling on-policy enforcement may lead to training instability. Use only when necessary and monitor training metrics closely.
:::

To disable enforcement:

```yaml
# Disable monotonicity enforcement and on-policy token ID correction
enforce_monotonicity: false
```

## Recommended Approaches

### For Models with Reasoning Traces

1. **Preferred**: Disable reasoning truncation and keep reasoning across all turns
2. **Alternative**: Use models without a reasoning component

### For Agents with Context Management

- Evaluate whether history modification is necessary for your use case
- If you must modify history, monitor training stability closely
- Consider importance sampling to handle off-policy data

## Related Topics

- {doc}`../../contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction` — Technical details on on-policy corrections
