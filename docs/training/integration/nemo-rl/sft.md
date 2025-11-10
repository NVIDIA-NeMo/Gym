(training-integration-nemo-rl-sft)=

# SFT Training

Train models with Supervised Fine-Tuning using high-quality NeMo Gym demonstrations.

```{contents}
:local:
:depth: 2
```

---

### When to Use SFT

**Supervised Fine-Tuning (SFT)** is ideal when:
- You have high-quality demonstration data
- Rewards are mostly high (≥0.90)
- You want the model to imitate specific behavior patterns

**Example use cases**: Instruction following, consistent tool usage, output formatting

:::{seealso}
**Understanding rollout data**: For details on the NeMo Gym rollout format that you'll be transforming, refer to:
- {doc}`../index` - Data format reference with complete rollout structure
- {doc}`../../../about/concepts/rollout-collection-fundamentals` - Conceptual overview of rollouts
:::

### Step 1: Transform Data for SFT

SFT expects conversation-format data. For tool-calling tasks, use OpenAI format:

```python
"""
Transform NeMo Gym rollouts to NeMo RL OpenAI SFT format.

Usage:
    python transform_gym_to_sft.py \
        --input rollouts.jsonl \
        --output sft_train.jsonl \
        --reward_threshold 0.95 \
        --format openai
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def gym_output_to_messages(gym_output: List[Dict], input_messages: List[Dict]) -> List[Dict]:
    """
    Convert NeMo Gym output format to OpenAI messages format.
    
    Args:
        gym_output: List of output items from NeMo Gym
        input_messages: Original input messages
    
    Returns:
        List of messages in OpenAI format
    """
    messages = []
    
    # Start with input messages (system + user)
    for msg in input_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append({"role": role, "content": content})
    
    # Process output items
    tool_calls = []
    for item in gym_output:
        item_type = item.get("type")
        
        if item_type == "function_call":
            # Collect tool call
            tool_call = {
                "name": item.get("name"),
                "arguments": json.loads(item.get("arguments", "{}"))
            }
            tool_calls.append(tool_call)
        
        elif item_type == "function_call_output":
            # Add assistant message with tool calls (if any)
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                })
                tool_calls = []
            
            # Add tool response
            messages.append({
                "role": "tool",
                "content": item.get("output", "")
            })
        
        elif item_type == "message":
            # Add any pending tool calls first
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                })
                tool_calls = []
            
            # Add assistant message
            content = item.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                text = content[0].get("text", "")
            elif isinstance(content, str):
                text = content
            else:
                text = ""
            
            messages.append({
                "role": "assistant",
                "content": text
            })
    
    return messages


def extract_tools(responses_create_params: Dict) -> List[Dict]:
    """Extract tool definitions from responses_create_params."""
    tools = responses_create_params.get("tools", [])
    
    # Convert to NeMo RL expected format if needed
    formatted_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            formatted_tools.append({
                "name": func.get("name"),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {})
            })
        else:
            formatted_tools.append(tool)
    
    return formatted_tools


def transform_rollout_to_openai(rollout: Dict) -> Dict:
    """Transform single rollout to OpenAI format."""
    input_messages = rollout["responses_create_params"]["input"]
    gym_output = rollout["output"]
    
    messages = gym_output_to_messages(gym_output, input_messages)
    tools = extract_tools(rollout["responses_create_params"])
    
    result = {"messages": messages}
    if tools:
        result["tools"] = tools
    
    return result


def transform_rollout_to_simple(rollout: Dict) -> Dict:
    """Transform to simple input/output format (no tool calling structure)."""
    # Extract input
    input_messages = rollout["responses_create_params"]["input"]
    input_text = ""
    for msg in reversed(input_messages):
        if msg.get("role") == "user":
            input_text = msg.get("content", "")
            break
    
    # Extract output
    output_text = ""
    for item in reversed(rollout["output"]):
        if item.get("type") == "message":
            content = item.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                output_text = content[0].get("text", "")
            elif isinstance(content, str):
                output_text = content
            break
    
    return {
        "input": input_text,
        "output": output_text
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transform NeMo Gym rollouts to NeMo RL SFT format"
    )
    parser.add_argument("--input", required=True, help="Input rollouts.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=0.95,
        help="Minimum reward to include (default: 0.95)",
    )
    parser.add_argument(
        "--format",
        choices=["openai", "simple"],
        default="openai",
        help="Output format: 'openai' for tool calling, 'simple' for text only",
    )
    
    args = parser.parse_args()
    
    # Load rollouts
    rollouts = []
    with open(args.input, "r") as f:
        for line in f:
            rollouts.append(json.loads(line))
    
    print(f"Loaded {len(rollouts)} rollouts from {args.input}")
    
    # Filter by reward threshold (SFT wants high quality only)
    filtered_rollouts = [
        r for r in rollouts if r.get("reward", 0.0) >= args.reward_threshold
    ]
    
    print(f"After filtering (reward >= {args.reward_threshold}): {len(filtered_rollouts)} rollouts")
    
    if len(filtered_rollouts) < 100:
        print("⚠️  Warning: Fewer than 100 examples. Consider lowering --reward_threshold")
    
    # Transform
    transformed = []
    transform_fn = (
        transform_rollout_to_openai if args.format == "openai" else transform_rollout_to_simple
    )
    
    for rollout in filtered_rollouts:
        try:
            transformed_data = transform_fn(rollout)
            transformed.append(transformed_data)
        except Exception as e:
            print(f"Warning: Failed to transform rollout: {e}")
            continue
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in transformed:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Wrote {len(transformed)} transformed rollouts to {args.output}")
    
    # Show example
    if transformed:
        print("\n--- Example Transformed Rollout ---")
        print(json.dumps(transformed[0], indent=2))


if __name__ == "__main__":
    main()
```

**Run transformation**:

```bash
# Transform high-quality rollouts for SFT (OpenAI format with tool calling)
python transform_gym_to_sft.py \
    --input rollouts.jsonl \
    --output sft_data/train.jsonl \
    --reward_threshold 0.95 \
    --format openai

# Or simple text format (no tool calling structure)
python transform_gym_to_sft.py \
    --input rollouts.jsonl \
    --output sft_data/train.jsonl \
    --reward_threshold 0.95 \
    --format simple
```

:::{tip}
**Tune the reward threshold**: The default `--reward_threshold 0.95` selects only the highest-quality demonstrations. Depending on your data distribution, you may want to adjust this:
- **0.95-1.0**: Very strict, ensures only near-perfect demonstrations
- **0.90-0.95**: Balanced, filters out poor attempts while keeping good examples
- **0.85-0.90**: More permissive, useful when high-reward rollouts are scarce

Check your rollout distribution first:
```bash
python -c "import json; rewards = [json.loads(l)['reward'] for l in open('rollouts.jsonl')]; import statistics; print(f'Mean: {statistics.mean(rewards):.2f}, Median: {statistics.median(rewards):.2f}')"
```
:::

**Inspect transformation output**:

Before starting training, verify the transformation worked correctly:

```bash
# View first transformed example
head -n 1 sft_data/train.jsonl | python -m json.tool

# Count examples
wc -l sft_data/train.jsonl
# Should show: <N> examples where N = rollouts with reward >= threshold
```

**Expected output structure**:
- **OpenAI format**: `{"messages": [...], "tools": [...]}`
- **Simple format**: `{"input": "...", "output": "..."}`

### Step 2: Configure NeMo RL for SFT

Create `sft_gym_data.yaml`:

```yaml
# Extend NeMo RL's base SFT config
defaults:
  - sft

# Configure for OpenAI format data
data:
  dataset_name: openai_format
  train_data_path: "sft_data/train.jsonl"
  val_data_path: "sft_data/val.jsonl"
  chat_key: "messages"
  tool_key: "tools"
  use_preserving_dataset: true  # Important for tool calling datasets

# Model configuration
policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  max_total_sequence_length: 1024
  train_global_batch_size: 128
  train_micro_batch_size: 4

sft:
  max_num_epochs: 3
  val_period: 100
  val_at_start: true

checkpointing:
  enabled: true
  checkpoint_dir: "results/gym_sft"
  metric_name: "val:loss"
  higher_is_better: false

logger:
  wandb_enabled: true
  wandb:
    project: "nemo-gym-sft"
    name: "gym-sft-run-1"
```

**For simple format** (no tool calling):

```yaml
data:
  dataset_name: ResponseDataset
  train_data_path: "sft_data/train.jsonl"
  val_data_path: "sft_data/val.jsonl"
  input_key: "input"
  output_key: "output"
```

### Step 3: Launch SFT Training

```bash
cd /path/to/nemo-rl

# Single GPU
uv run python examples/run_sft.py \
    --config sft_gym_data.yaml

# Multi-GPU (8 GPUs)
uv run python examples/run_sft.py \
    --config sft_gym_data.yaml \
    cluster.gpus_per_node=8

# Multi-node
COMMAND="uv run python examples/run_sft.py --config sft_gym_data.yaml cluster.num_nodes=2" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch --nodes=2 --gres=gpu:8 ray.sub
```

---

## Troubleshooting

### Transformation Issues

**"Fewer than 100 examples" warning**
- **Cause**: Not enough rollouts meet the reward threshold
- **Solution**: Lower `--reward_threshold` (try 0.90 or 0.85) or collect more rollouts

**Tool calls missing in output**
- **Cause**: Rollout format mismatch or missing tool definitions
- **Solution**: Verify your rollouts have `responses_create_params.tools` field:
  ```bash
  head -n 1 rollouts.jsonl | python -m json.tool | grep -A 5 '"tools"'
  ```

**"KeyError: 'messages'" during training**
- **Cause**: Transformation script didn't run correctly
- **Solution**: Check transformed file structure matches expected format (see inspection commands above)

### Training Issues

**"Tool schema corruption" warning**
- **Cause**: Using standard dataset loader with heterogeneous tool schemas
- **Solution**: Set `use_preserving_dataset: true` in your config (see Step 2)

**Validation loss not improving**
- **Cause**: Dataset too small or reward threshold too strict
- **Solution**: 
  - Check dataset size: `wc -l sft_data/train.jsonl` (aim for 1,000+ examples)
  - Review reward distribution and adjust threshold if needed
  - Consider using {doc}`grpo` instead if you have mixed-quality data

**Out of memory errors**
- **Cause**: Sequence length or batch size too large
- **Solution**: Reduce `max_total_sequence_length` or `train_micro_batch_size` in config

:::{seealso}
**More training algorithms**: If SFT isn't the right fit for your data:
- {doc}`grpo` - For mixed-quality rollouts with automatic verification
- {doc}`dpo` - For preference pairs (chosen vs. rejected outputs)
:::

---

