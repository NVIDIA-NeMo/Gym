(env-rlhf-reward-models)=

# RLHF with Reward Models

Use trained reward models as environments for GRPO training to optimize against learned human preferences.

:::{card}

**Goal**: Train with reward models instead of rule-based verification.

^^^

**In this tutorial, you will**:

1. Train or load a Bradley-Terry reward model
2. Configure the reward model environment
3. Run GRPO training with learned preferences

:::

:::{button-ref} multi-turn
:color: secondary
:outline:
:ref-type: doc

← Previous: Multi-Turn Environments
:::

:::{tip}
**When to use reward models vs. LLM-as-judge:**

| Approach | Best for | Trade-off |
|----------|----------|-----------|
| Reward model | High-throughput training, learned preferences | Requires preference data and training |
| LLM-as-judge | Flexible evaluation, no training required | Higher latency, API costs |
| Hybrid | Best of both | Complex setup |

For semantic equivalence checking, see {doc}`llm-as-judge`.
:::

---

## How It Works

Reward models score conversation quality based on learned human preferences:

```text
Model Response → Tokenize → Reward Model → Score → Training Signal
                               │
                               ▼
                    Bradley-Terry classifier
                    (single scalar output)
```

**Key difference from LLM-as-judge**: Reward models are trained classifiers that output a single score, not generative models that reason about equivalence.

## Implementation

Reward model environments are implemented in **NeMo RL**, not NeMo Gym resources servers. The `RewardModelEnvironment` class wraps trained reward models as RL environments.

### Supported Models

Currently supported reward model type:

| Type | Description | Output |
|------|-------------|--------|
| `bradley_terry` | Binary preference classifier | Single scalar score |

```{note}
Bradley-Terry models are trained on pairwise preference data where annotators choose between two completions. The model learns to predict which response humans prefer.
```

### Core Components

The reward model environment has three main functions:

1. **Preprocessing**: Tokenizes conversation logs with proper formatting
2. **Scoring**: Runs the reward model to compute scores
3. **Environment interface**: Returns rewards compatible with GRPO training

```python
# From nemo_rl/environments/reward_model_environment.py

def step(
    self,
    message_logs: List[LLMMessageLogType],
    env_infos: List[Dict[str, Any]],
) -> EnvironmentReturn:
    # Preprocess the message logs
    reward_data = self.preprocess_data(message_logs)

    # Score the message logs
    rewards = self.reward_model_policy.score(reward_data)["scores"]

    # All episodes terminate after one step in reward model environment
    terminateds = [True] * len(message_logs)

    return EnvironmentReturn(
        observations=observations,
        metadata=metadata,
        next_stop_strings=next_stop_strings,
        rewards=rewards.cpu(),
        terminateds=torch.tensor(terminateds, dtype=torch.bool).cpu(),
        answers=answers,
    )
```

## Quick Start

### 1. Train a Reward Model

First, train a Bradley-Terry reward model on preference data:

```bash
# From NeMo RL directory
uv run examples/run_rm.py --config examples/configs/rm.yaml
```

**Supported datasets**:

- [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) — General helpfulness preferences
- [Tulu3Preference](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-math-grade) — Diverse preference data
- Custom JSONL with `context`, `completions`, and `rank` fields

### 2. Configure the Environment

Create a GRPO config that uses your trained reward model:

```yaml
# grpo_with_rm.yaml
env:
  reward_model:
    model_name: "path/to/your/trained/reward/model"
    # Or use a pretrained model from HuggingFace:
    # model_name: "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    tokenizer:
      name: ${env.reward_model.model_name}
    precision: "bfloat16"
    batch_size: ${policy.train_micro_batch_size}
    resources:
      gpus_per_node: 1
      num_nodes: 1
    dtensor_cfg:
      enabled: true
      tensor_parallel_size: 1
    reward_model_cfg:
      enabled: true
      reward_model_type: "bradley_terry"
    dynamic_batching:
      enabled: false
    sequence_packing:
      enabled: false
```

### 3. Run GRPO Training

```bash
# From NeMo RL directory
uv run examples/run_grpo_rm.py --config examples/configs/grpo_rm_1B.yaml

# With custom reward model path
uv run examples/run_grpo_rm.py \
    --config examples/configs/grpo_rm_1B.yaml \
    env.reward_model.model_name=path/to/your/model
```

## Configuration Reference

### Required Settings

| Option | Type | Description |
|--------|------|-------------|
| `model_name` | str | Path to trained reward model or HuggingFace model ID |
| `tokenizer.name` | str | Tokenizer (usually same as model) |
| `reward_model_cfg.enabled` | bool | Must be `true` |
| `reward_model_cfg.reward_model_type` | str | Only `bradley_terry` supported |
| `dtensor_cfg.enabled` | bool | Must be `true` (Megatron path not yet supported) |

### Resource Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `resources.gpus_per_node` | int | 1 | GPUs per node for reward model |
| `resources.num_nodes` | int | 1 | Number of nodes |
| `precision` | str | `bfloat16` | Model precision |
| `batch_size` | int | varies | Batch size for scoring |

### Current Limitations

```{warning}
The reward model environment has specific requirements:
- DTensor backend required (`dtensor_cfg.enabled: true`)
- Megatron backend not yet supported ([tracking issue](https://github.com/NVIDIA-NeMo/RL/issues/1154))
- Dynamic batching disabled
- Sequence packing disabled
- CPU offload disabled
- Activation checkpointing disabled
```

## Preference Data Format

To train a reward model, prepare data in this format:

```json
{
  "context": [
    {"role": "user", "content": "What's the capital of France?"}
  ],
  "completions": [
    {
      "rank": 0,
      "completion": [{"role": "assistant", "content": "The capital of France is Paris."}]
    },
    {
      "rank": 1,
      "completion": [{"role": "assistant", "content": "France is a country in Europe."}]
    }
  ]
}
```

**Key fields**:

- `context`: Conversation history up to the point of completion
- `completions`: List of completions with ranks (lower = better)
- `rank`: 0 = preferred, 1 = rejected

## Pretrained Reward Models

Several pretrained reward models work with NeMo RL:

| Model | Size | License |
|-------|------|---------|
| [Skywork/Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B) | 0.6B | Apache 2.0 |
| [Skywork/Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B) | 8B | Apache 2.0 |

```{note}
Verify the license of any reward model permits your use case before deployment.
```

## Metrics

The reward model environment reports these metrics during training:

| Metric | Description |
|--------|-------------|
| `reward_model_env/num_samples` | Samples processed |
| `reward_model_env/mean_reward` | Average reward across batch |
| `reward_model_env/std_reward` | Reward standard deviation |
| `reward_model_env/min_reward` | Minimum reward in batch |
| `reward_model_env/max_reward` | Maximum reward in batch |

## Alternatives

| Approach | Use When | Implementation |
|----------|----------|----------------|
| **Reward model** | Training at scale with preference data | `RewardModelEnvironment` |
| **LLM-as-judge** | Flexible evaluation, no training needed | {doc}`llm-as-judge` |
| **Rule-based** | Deterministic criteria (exact match, regex) | {doc}`creating-training-environment` |
| **Hybrid** | Library check + LLM fallback | `math_with_judge` resources server |

## Next Steps

- {doc}`llm-as-judge` — LLM-based verification (no training required)
- {doc}`creating-training-environment` — Build custom verification logic
- {ref}`training-nemo-rl-grpo-index` — Train with verified rewards
