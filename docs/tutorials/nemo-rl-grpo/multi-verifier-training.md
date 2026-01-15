(training-nemo-rl-grpo-multi-verifier)=
# Train with Multiple Verifiers

Combine multiple verification strategies for more robust reward signals during GRPO training.

**Goal**: Configure NeMo Gym to use multiple resource servers or reward functions in a single training run.

**Prerequisites**:
- Completed {doc}`/get-started/detailed-setup`
- Familiar with {doc}`/resources-server/index`

---

## Quick Start

Combine two resource servers (math + search) for training:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[${config_paths}]"
```

The `+` prefix is [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) for appending to config.

---

## When to Use

Combine verifiers when:

- **Single verifier is unreliable**: LLM judges have variance; rule-based verifiers miss edge cases
- **Task has distinct success criteria**: Correctness + format + efficiency
- **You want balanced signals**: Combine cheap/fast rule-based with expensive/accurate model-based

**Example**: Math problems benefit from combining:
1. Fast rule-based format checking (did the model use `<think>` and `<answer>` tags?)
2. Mathematical equivalence checking (is the answer mathematically correct?)
3. LLM judge for edge cases (are `π` and `3.14159` equivalent in this context?)

---

## Two Approaches

NeMo Gym supports two complementary approaches:

| Approach | Where | Use Case |
|----------|-------|----------|
| **Distinct Resource Servers** | NeMo Gym | Different verification domains (math + search + code) |
| **Reward Function Combination** | NeMo RL | Weighted blend within one domain |

---

## Approach 1: Distinct Resource Servers (NeMo Gym)

Combine resource servers by adding their YAML configs together. Each server independently verifies responses for its domain.

:::{seealso}
{doc}`/resources-server/index` explains resource server concepts.
:::

### Configuration

Chain config files in `config_paths`:

```bash
# Individual servers
config_paths_math="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"

config_paths_search="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/google_search/configs/google_search.yaml"

# Combined: add all YAML paths together
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### How It Works

```text
Agent Response
    ├── Math Resources Server → verify() → reward_math
    └── Search Resources Server → verify() → reward_search
                    │
                    ▼
        Training uses task-specific reward
        (each sample routed to its server)
```

Each training sample routes to the appropriate resource server based on its task type.

### Data Preparation

Prepare data for all configured servers:

```bash
config_paths="resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/multi_verifier \
    +mode=train_preparation \
    +should_download=true
```

The collated training data includes samples from all configured environments.

### Scaling Considerations

- Each resource server runs as a separate process
- For high-throughput training, profile servers using {doc}`/resources-server/profile`
- If one server fails, training stops—ensure all servers are reliable before production runs

---

## Approach 2: Reward Function Combination (NeMo RL)

Combine reward functions within a single environment using weighted averaging. Configured in the NeMo RL training YAML.

### Configuration

Define `reward_functions` with weights:

```yaml
env:
  geometry3k:
    num_workers: 8
    reward_functions:
      - name: format
        weight: 0.1
      - name: math_expr
        weight: 0.9
```

### Available Reward Functions

| Name | Description | Returns |
|------|-------------|---------|
| `format` | Checks `<think>` and `<answer>` tags | `(0.0-1.0, None)` |
| `exact_alnum` | Exact alphanumeric match (case-insensitive) | `(0.0 or 1.0, bool)` |
| `math_expr` | Mathematical expression equivalence | `(0.0 or 1.0, bool)` |
| `bbox_giou` | Bounding box GIoU for visual tasks | `(float, bool)` |

**Source**: `nemo_rl/environments/vlm_environment.py:71-78`

### How Weights Combine

From `nemo_rl/environments/rewards.py:145-173`:

```python
def combine_reward_functions(reward_functions):
    """Combine reward functions with normalized weights."""
    weights = [weight for _, weight in reward_functions]
    weights = np.array(weights) / np.sum(weights)  # Auto-normalize to 1

    def combined_reward_func(ground_truth, response):
        rewards = [func(ground_truth, response)[0] for func, _ in reward_functions]
        is_correct = all(
            r[1] for func, _ in reward_functions
            if (r := func(ground_truth, response))[1] is not None
        )
        return np.sum(np.array(rewards) * weights), is_correct

    return combined_reward_func
```

**Key behavior**:
- Weights **auto-normalize** to sum to 1.0 (no manual normalization needed)
- `is_correct` requires **all** functions with correctness signals to pass
- Functions returning `None` for correctness (like `format`) don't affect correctness checks

### Example: VLM Training

From `examples/configs/vlm_grpo_3B.yaml`:

```yaml
env:
  clevr-cogent:
    num_workers: 8
    reward_functions:
      - name: format
        weight: 0.2      # 20% of reward
      - name: exact_alnum
        weight: 0.8      # 80% of reward

  refcoco:
    num_workers: 8
    reward_functions:
      - name: format
        weight: 0.1
      - name: bbox_giou
        weight: 0.9
        kwargs:
          giou_penalty_thres: 0.5
```

---

## Choosing an Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Different task types in training data | Distinct Resource Servers |
| Same task, distinct quality signals | Reward Function Combination |
| Both scenarios | Use both approaches together |

**Example combined setup**:
- Distinct resource servers for math, code, and search tasks
- Each server internally combines format + correctness rewards

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| All rewards are 0 | Verifier can't parse response format | Check response tags match expected format |
| Weights don't sum to 1 | Expected behavior | Weights auto-normalize |
| `is_correct` always False | Any contributing function fails | Check individual function outputs |
| Server not receiving requests | Task routing mismatch | Verify dataset `task` field matches server |

### Debugging Reward Signals

Test reward functions directly:

```python
from nemo_rl.environments.rewards import (
    format_reward,
    math_expression_reward,
    combine_reward_functions,
)

response = "<think>Let me solve...</think><answer>42</answer>"
ground_truth = "42"

# Test individually
print(f"Format: {format_reward(ground_truth, response)}")
print(f"Math: {math_expression_reward(ground_truth, response)}")

# Test combined
combined = combine_reward_functions([
    (format_reward, 0.2),
    (math_expression_reward, 0.8),
])
print(f"Combined: {combined(ground_truth, response)}")
```

**Expected output**:
```text
Format: (1.0, None)
Math: (1.0, True)
Combined: (1.0, True)
```

---

## Related Topics

- {doc}`/about/concepts/task-verification` — How verification drives training
- {doc}`/resources-server/index` — Resource server concepts
- {doc}`/resources-server/profile` — Profile server performance
- [NeMo RL GRPO Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md) — GRPO training configuration
