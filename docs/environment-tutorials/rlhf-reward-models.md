(env-rlhf-reward-models)=
# RLHF with Reward Models

```{warning}
This article was generated and has not been reviewed. Content may change.
```

Integrate learned reward models for Reinforcement Learning from Human Feedback (RLHF).

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
45-60 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`/get-started/detailed-setup`
- Access to a reward model (Hugging Face, custom trained)

:::

::::

---

## What is RLHF?

RLHF uses a learned reward model trained on human preferences:

- Reward model predicts human preference scores
- Training optimizes model outputs for higher reward
- Enables alignment with human values and intent

```text
Agent Response → Reward Model → Reward Score (continuous) → Training Signal
```

## When to Use

Choose reward models when human preferences are central to your task:

| Approach | Best For | Trade-offs |
|----------|----------|------------|
| **Reward Model** | Learned preferences, subjective quality, alignment | Requires training data; may have distribution shift |
| **LLM-as-Judge** | Equivalence checking, open-ended evaluation | API costs; prompt engineering required |
| **Programmatic** | Math, code execution, exact match | Limited to verifiable tasks |

Use RLHF reward models when:

- You have human preference data (pairwise comparisons or ratings)
- Tasks involve subjective quality (helpfulness, safety, tone)
- You need consistent, fast inference at scale
- Hand-crafted rules can't capture the evaluation criteria

## Quick Start

### 1. Configure the Reward Model Server

Create a configuration file for your reward model resources server:

```yaml
reward_model_server:
  resources_servers:
    reward_model:
      entrypoint: app.py
      model_path: /path/to/reward_model
      device: cuda
```

### 2. Implement the Resources Server

```python
class RewardModelResourcesServer(SimpleResourcesServer):
    def model_post_init(self, context):
        # Load reward model
        self.reward_model = load_reward_model(self.config.model_path)
    
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Extract response text
        response_text = extract_response(body.response)
        
        # Get reward model score
        reward = self.reward_model.score(
            prompt=body.responses_create_params.input,
            response=response_text
        )
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

### 3. Start the Servers

```bash
config_paths="resources_servers/reward_model/configs/reward_model.yaml"

ng_run "+config_paths=[$config_paths]"
```

### 4. Collect Rollouts

```bash
ng_collect_rollouts \
    +agent_name=reward_model_simple_agent \
    +input_jsonl_fpath=data/train.jsonl \
    +output_jsonl_fpath=data/rollouts.jsonl
```

## Configuration Reference

### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_path` | str | required | Path to reward model weights |
| `device` | str | `cuda` | Device for inference (`cuda`, `cpu`) |
| `batch_size` | int | `8` | Batch size for reward inference |
| `max_length` | int | `2048` | Maximum sequence length |

### Normalization Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `normalize_rewards` | bool | `false` | Apply reward normalization |
| `reward_mean` | float | `0.0` | Running mean for normalization |
| `reward_std` | float | `1.0` | Running std for normalization |

## Key RLHF Concepts

Understanding these concepts is essential for effective RLHF training.

::::{dropdown} Reward Normalization
:icon: graph

Raw reward model outputs may have arbitrary scale and distribution. Normalize rewards to stabilize training:

```python
def normalize_reward(self, raw_reward: float) -> float:
    # Running statistics normalization
    normalized = (raw_reward - self.running_mean) / (self.running_std + 1e-8)
    # Clip to prevent extreme values
    return max(-5.0, min(5.0, normalized))
```

**When to normalize:**

- Reward distributions shift during training
- Combining multiple reward signals
- Reward scale varies across data subsets

::::

::::{dropdown} KL Divergence Constraints
:icon: shield

Prevent the policy from diverging too far from the reference model:

```python
# PPO/GRPO objective with KL penalty
reward_with_kl = reward - beta * kl_divergence(policy, reference)
```

| KL Coefficient (β) | Effect |
|--------------------|--------|
| Low (0.01-0.05) | More optimization, risk of reward hacking |
| Medium (0.1-0.2) | Balanced exploration and stability |
| High (0.5+) | Conservative updates, slower learning |

::::

::::{dropdown} Reward Hacking Mitigation
:icon: alert

Models may exploit reward model weaknesses. Common mitigations:

1. **Diverse training data**: Include adversarial examples in reward model training
2. **Ensemble rewards**: Average multiple reward models
3. **Length penalties**: Prevent verbose responses from gaming rewards
4. **Human spot-checks**: Periodically validate high-reward outputs

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    raw_reward = self.reward_model.score(prompt, response)
    
    # Length penalty to discourage verbosity gaming
    length_penalty = max(0, len(response) - 500) * 0.001
    
    reward = raw_reward - length_penalty
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

::::

## Reward Model Types

:::::{tab-set}

::::{tab-item} Preference-Based

Trained on pairwise comparisons (response A vs response B):

```python
# Outputs score in [0, 1] range
# Higher = more preferred
reward = reward_model.score(prompt, response)
```

**Common architectures:**

- Bradley-Terry models
- Reward models fine-tuned from LLMs (e.g., removing the LM head, adding a scalar head)

::::

::::{tab-item} Scalar Reward

Trained to predict absolute quality scores:

```python
# Outputs continuous value (may need normalization)
reward = reward_model.predict(prompt, response)
```

**Use cases:**

- Direct quality ratings (1-5 stars)
- Absolute safety scores
- Multi-attribute scoring

::::

:::::

## Input Data Format

### Training Data

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Explain quantum computing simply."}
    ]
  }
}
```

### With Expected Behavior (Optional)

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing simply."}
    ]
  },
  "reward_threshold": 0.7
}
```

## Supported Models

:::::{tab-set}

::::{tab-item} Hugging Face Reward Models

Load reward models from Hugging Face Hub:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class HFRewardModelServer(SimpleResourcesServer):
    def model_post_init(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def score(self, prompt: str, response: str) -> float:
        inputs = self.tokenizer(
            prompt + response,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.logits[0].item()
```

::::

::::{tab-item} Custom Reward Models

For custom architectures, implement the scoring interface:

```python
class CustomRewardModel:
    def __init__(self, model_path: str):
        self.model = load_custom_model(model_path)
    
    def score(self, prompt: str, response: str) -> float:
        # Your custom scoring logic
        return self.model.forward(prompt, response)
```

::::

:::::

## Example

<!-- TODO: Add complete RLHF example with training integration -->

For reward model integration patterns, see:

- `resources_servers/reward_model/` — Basic reward model server

## Next Steps

- Compare with {doc}`llm-as-judge` for open-ended evaluation
- Learn about {doc}`creating-training-environment` for reward design
- Start training with {ref}`training-nemo-rl-grpo-index`
