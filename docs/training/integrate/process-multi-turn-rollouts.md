(integrate-process-multi-turn-rollouts)=

# Process Multi-Turn Rollouts

Convert Gym rollout results into training-ready token sequences with correct alignment across turns.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
25 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`connect-gym-to-training`
- Gym rollouts being collected
- Understanding of your tokenizer behavior

:::

::::

---

## Why This Matters

Multi-turn rollouts involve multiple model calls within a single trajectory. Between calls, text is detokenized and re-tokenized, which can cause **token alignment issues**:

```text
Turn 1: Model generates token IDs [10, 11, 12] → detokenized to "Hello"
Turn 2: "Hello" is re-tokenized → might become [10, 11] or [13] (different!)
```

If you train on misaligned tokens, your gradients will be incorrect. This guide shows you how to handle this correctly.

:::{seealso}
For a deeper explanation of why this happens, refer to {doc}`/about/concepts/training-integration-architecture`.
:::

---

## Understand the Rollout Structure

A multi-turn rollout contains multiple model outputs. Here's a typical structure:

```python
# Multi-step rollout example
rollout = {
    "response": {
        "output": [
            # Turn 1: User message (not from model)
            {"type": "message", "role": "user", "content": "What's 2+2?"},
            
            # Turn 1: Model reasoning + answer
            {
                "type": "message",
                "role": "assistant",
                "content": "<think>Simple addition</think>The answer is 4.",
                "prompt_token_ids": [1, 2, 3, 4, 5],      # Full context
                "generation_token_ids": [10, 11, 12],     # What model generated
                "generation_log_probs": [-0.5, -0.3, -0.1],
            },
            
            # Turn 1: Tool call (if applicable)
            {
                "type": "function_call",
                "name": "verify_answer",
                "arguments": "{\"answer\": 4}",
                "prompt_token_ids": [1, 2, 3, 4, 5, 10, 11, 12],
                "generation_token_ids": [20, 21, 22],
                "generation_log_probs": [-0.2, -0.4, -0.3],
            },
            
            # Tool response (not from model)
            {"type": "function_call_output", "output": "Correct!"},
            
            # Turn 2: Model's final response
            {
                "type": "message",
                "role": "assistant", 
                "content": "I verified the answer is correct.",
                "prompt_token_ids": [1, 2, 3, ..., 25],  # Full history
                "generation_token_ids": [30, 31, 32],
                "generation_log_probs": [-0.1, -0.2, -0.1],
            },
        ],
    },
    "reward": 1.0,
}
```

**Key insight**: Only items with `generation_token_ids` are trainable (model outputs).

---

## Extract Trainable Sequences

Filter rollout outputs to extract only trainable model generations:

```python
from transformers import PreTrainedTokenizerBase


def extract_trainable_turns(rollout: dict) -> list[dict]:
    """
    Extract turns that have token IDs (trainable model outputs).
    
    Args:
        rollout: Gym rollout result
        
    Returns:
        List of trainable turn dicts with token_ids and log_probs
    """
    trainable_turns = []
    
    for output_item in rollout["response"]["output"]:
        # Only process items with generation token IDs
        if "generation_token_ids" not in output_item:
            continue
            
        trainable_turns.append({
            "prompt_token_ids": output_item["prompt_token_ids"],
            "generation_token_ids": output_item["generation_token_ids"],
            "generation_log_probs": output_item["generation_log_probs"],
        })
    
    return trainable_turns
```

---

## Validate Token Continuity

The critical step: verify that token sequences are contiguous across turns. If turn N's prompt doesn't start with turn N-1's full sequence, you have an alignment problem.

```python
def validate_token_continuity(trainable_turns: list[dict]) -> bool:
    """
    Validate that token sequences are contiguous across turns.
    
    Args:
        trainable_turns: List of turns with token IDs
        
    Returns:
        True if sequences are contiguous, raises AssertionError otherwise
    """
    seen_token_ids = []
    
    for i, turn in enumerate(trainable_turns):
        prompt_ids = turn["prompt_token_ids"]
        generation_ids = turn["generation_token_ids"]
        
        # Check that this turn's prompt starts with all previously seen tokens
        if seen_token_ids:
            prefix = prompt_ids[:len(seen_token_ids)]
            
            assert prefix == seen_token_ids, f"""
Token discontinuity at turn {i}!

Expected prefix: {seen_token_ids[:20]}... (len={len(seen_token_ids)})
Actual prefix:   {prefix[:20]}... (len={len(prefix)})

This usually means:
1. Tokenizer merged tokens differently on re-tokenization
2. Chat template changed between turns
3. Context was truncated

Refer to docs for debugging: /about/concepts/training-integration-architecture
"""
        
        # Accumulate tokens for next turn's validation
        seen_token_ids = prompt_ids + generation_ids
    
    return True
```

---

## Build Training-Ready Message Log

Convert validated turns into a format suitable for your training framework:

```python
import torch


def build_message_log(
    trainable_turns: list[dict],
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict]:
    """
    Build a training-ready message log from validated turns.
    
    Args:
        trainable_turns: Validated turns with token IDs
        tokenizer: Tokenizer for debugging/logging
        
    Returns:
        List of messages with token tensors for training
    """
    message_log = []
    seen_token_ids = []
    
    for turn in trainable_turns:
        prompt_ids = turn["prompt_token_ids"]
        generation_ids = turn["generation_token_ids"]
        log_probs = turn["generation_log_probs"]
        
        # Extract NEW prompt tokens (not seen before)
        new_prompt_ids = prompt_ids[len(seen_token_ids):]
        
        # Add prompt segment (user/context - not trained on)
        if new_prompt_ids:
            message_log.append({
                "role": "user",  # Or "context" depending on your framework
                "content": tokenizer.decode(new_prompt_ids),
                "token_ids": torch.tensor(new_prompt_ids),
                "trainable": False,
            })
        
        # Add generation segment (assistant - trained on)
        message_log.append({
            "role": "assistant",
            "content": tokenizer.decode(generation_ids),
            "token_ids": torch.tensor(generation_ids),
            "log_probs": torch.tensor(log_probs),
            "trainable": True,
        })
        
        # Update seen tokens
        seen_token_ids = prompt_ids + generation_ids
    
    return message_log
```

---

## Complete Processing Pipeline

Put it all together into a single processing function:

```python
def process_rollout_for_training(
    rollout: dict,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """
    Process a Gym rollout into training-ready format.
    
    Args:
        rollout: Raw Gym rollout result
        tokenizer: Tokenizer for the model
        
    Returns:
        Training-ready dict with message_log and reward
    """
    # 1. Extract trainable turns
    trainable_turns = extract_trainable_turns(rollout)
    
    if not trainable_turns:
        # No model outputs to train on
        return None
    
    # 2. Validate token continuity
    validate_token_continuity(trainable_turns)
    
    # 3. Build message log
    message_log = build_message_log(trainable_turns, tokenizer)
    
    # 4. Package for training
    return {
        "message_log": message_log,
        "reward": rollout["reward"],
        "num_turns": len(trainable_turns),
        # Keep full result for logging/debugging
        "full_rollout": rollout,
    }


# Process a batch of rollouts
def process_rollouts(
    rollouts: list[dict],
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict]:
    """Process a batch of rollouts for training."""
    processed = []
    
    for rollout in rollouts:
        try:
            result = process_rollout_for_training(rollout, tokenizer)
            if result:
                processed.append(result)
        except AssertionError as e:
            # Log discontinuity errors but continue
            print(f"Skipping rollout due to token discontinuity: {e}")
            continue
    
    return processed
```

---

## Handle Common Edge Cases

### Truncated Context

Long conversations may be truncated by the model's context window:

```python
def handle_truncation(
    trainable_turns: list[dict],
    max_context_length: int,
) -> list[dict]:
    """
    Handle cases where context was truncated.
    
    If early context was truncated, we can only train on turns
    after the truncation point.
    """
    # Find first turn where prompt matches expected length
    for i, turn in enumerate(trainable_turns):
        if len(turn["prompt_token_ids"]) < max_context_length:
            # This turn wasn't truncated, start here
            return trainable_turns[i:]
    
    # All turns truncated - skip this rollout
    return []
```

### Special Tokens

Some tokenizers add special tokens differently based on context:

```python
def strip_special_tokens(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Remove BOS/EOS tokens that might cause alignment issues."""
    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }
    return [t for t in token_ids if t not in special_ids]
```

---

## Verify Your Processing

Test with a sample rollout:

```python
# Test the processing pipeline
sample_rollout = await gym.collect_rollouts([sample_example]).__anext__()

processed = process_rollout_for_training(sample_rollout, tokenizer)

print(f"Turns: {processed['num_turns']}")
print(f"Reward: {processed['reward']}")

for msg in processed["message_log"]:
    print(f"  {msg['role']}: {len(msg['token_ids'])} tokens, trainable={msg['trainable']}")
```

**✅ Success**:

```text
Turns: 3
Reward: 1.0
  user: 15 tokens, trainable=False
  assistant: 23 tokens, trainable=True
  user: 8 tokens, trainable=False
  assistant: 45 tokens, trainable=True
  user: 12 tokens, trainable=False
  assistant: 18 tokens, trainable=True
```

---

## Troubleshooting

### Token discontinuity errors

**Symptom**: `AssertionError: Token discontinuity at turn N`

**Causes**:
1. **Tokenizer merging**: Some tokenizers merge adjacent tokens differently based on surrounding context
2. **Chat template changes**: Template may add different tokens based on conversation state
3. **Truncation**: Earlier context was truncated to fit context window

**Debug steps**:
```python
# Print the actual vs expected tokens
print("Expected:", tokenizer.decode(seen_token_ids[-20:]))
print("Actual:", tokenizer.decode(prompt_ids[:20]))
```

### Empty trainable turns

**Symptom**: `process_rollout_for_training` returns `None`

**Fix**: Ensure your rollouts include model outputs with `generation_token_ids`. Check that your resource server isn't filtering these fields.

---

## Next Step

Your rollouts are now training-ready. Next, validate that your entire integration works correctly.

:::{button-ref} validate-integration
:color: primary
:outline:

Next: Validate Your Integration →
:::

---

## Reference

- {doc}`/about/concepts/training-integration-architecture` — Token alignment deep-dive
- `nemo_rl/environments/penguin.py:140-198` — NeMo RL implementation

