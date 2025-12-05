(training-format-specification)=

# Format Specification

Schema definitions for SFT, DPO, and RL training data formats.

---

## SFT Format

**Purpose**: Train models to follow successful agent interaction patterns.

### Schema

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### Example

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "content": "Temperature: 22°C, sunny"},
    {"role": "assistant", "content": "The weather in Paris is 22°C and sunny."}
  ]
}
```

### Fields

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Required
  - Description
* - `messages`
  - Yes
  - Array of conversation turns following OpenAI chat format
* - `reward`
  - No
  - Quality score from verification (0.0–1.0), useful for filtering
* - `task_type`
  - No
  - Category label for balancing datasets
```

### Message Roles

- **user**: Human input or task prompt
- **assistant**: Model response (may include `tool_calls`)
- **tool**: Tool execution result (follows a `tool_calls` message)
- **system**: System prompt (if present, must be first)

---

## DPO Format

**Purpose**: Train models to prefer better responses over worse ones.

### Schema

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

### Example

```json
{
  "prompt": [{"role": "user", "content": "Solve: 2x + 5 = 13"}],
  "chosen": [
    {"role": "assistant", "content": "I'll solve step by step:\n2x + 5 = 13\n2x = 8\nx = 4"}
  ],
  "rejected": [
    {"role": "assistant", "content": "The answer is x = 3"}
  ]
}
```

### Fields

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Required
  - Description
* - `prompt`
  - Yes
  - Conversation context up to the point of divergence
* - `chosen`
  - Yes
  - The preferred response (higher reward)
* - `rejected`
  - Yes
  - The less preferred response (lower reward)
* - `quality_difference`
  - No
  - Reward delta between chosen and rejected
```

### Requirements

- **Minimum quality difference**: Pairs should have meaningful reward differences (recommend ≥0.1)
- **Same prompt**: Both responses must be for the identical prompt
- **2 rollouts per task**: DPO requires exactly 2 rollouts per prompt to form pairs

---

## RL Format (GRPO)

**Purpose**: On-policy reinforcement learning with Gym as the environment.

RL training uses Gym directly during the training loop. Data is prepared using `ng_prepare_data` which adds routing metadata.

### Schema (Input)

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "..."}],
    "model": "...",
    "tools": [...]
  },
  "agent_ref": "agent_name"
}
```

### Fields

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Field
  - Required
  - Description
* - `responses_create_params`
  - Yes
  - OpenAI Responses API compatible request
* - `agent_ref`
  - Yes
  - Agent server that handles this example (added by `ng_prepare_data`)
```

Refer to {doc}`prepare-data` for details on preparing RL training data.

---

## Rollout Output Format

When you collect rollouts with `ng_collect_rollouts`, each rollout contains:

```json
{
  "responses_create_params": {...},
  "output": [...],
  "reward": 0.85,
  "metadata": {...}
}
```

### Fields

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Description
  - Used For
* - `responses_create_params`
  - Original request parameters
  - Reconstructing the prompt
* - `output`
  - Complete conversation (messages array)
  - SFT training data
* - `reward`
  - Verification score (0.0–1.0)
  - Filtering, DPO pair creation
* - `metadata`
  - Task-specific metadata
  - Balancing, analysis
```

---

## Converting Rollouts to Training Formats

### Rollout → SFT

Extract the `output` field as `messages`:

```python
sft_example = {"messages": rollout["output"]}
```

### Rollout → DPO

Group rollouts by prompt, pair by reward difference:

```python
# For two rollouts with same prompt
if rollout_1["reward"] > rollout_2["reward"]:
    chosen, rejected = rollout_1["output"], rollout_2["output"]
else:
    chosen, rejected = rollout_2["output"], rollout_1["output"]

dpo_example = {
    "prompt": rollout_1["responses_create_params"]["input"],
    "chosen": chosen,
    "rejected": rejected
}
```

Refer to {doc}`/tutorials/integrate-training-frameworks/train-with-trl` for a complete tutorial.

