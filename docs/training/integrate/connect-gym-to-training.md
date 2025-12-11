---
description: "Integrate Gym's rollout collection into your custom RL training pipeline"
categories: ["how-to-guides"]
tags: ["training-loop", "rollout-collection", "integration", "async", "rl-training"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
---

(integrate-connect-gym-to-training)=

# Connect Gym to Your Training Loop

Integrate Gym's rollout collection into your custom RL training pipeline to receive token IDs, log probabilities, and rewards for policy training.

## How It Works

Gym provides a `RolloutCollectionHelper` that coordinates rollout collection across your training cluster:

1. Your training script sends batches of prompts to Gym
2. Gym orchestrates generation through your policy endpoint
3. Agents execute multi-turn interactions and compute rewards
4. Results return with token-level data ready for gradient computation

## Before You Start

**Prerequisites**:

- Completed {doc}`expose-openai-endpoint`
- Working OpenAI-compatible endpoint
- Existing training loop that needs rollouts

**Installation**:

```bash
pip install nemo-gym
```

:::{dropdown} Add as Optional Dependency
:icon: package

```toml
# pyproject.toml
[project.optional-dependencies]
gym = ["nemo-gym>=0.1.0"]
```

:::

---

## Quick Start

### 1. Create Configuration

Create `gym_env.yaml` in your training directory:

```yaml
# gym_env.yaml - Minimal config for training integration
policy_model_name: "${POLICY_MODEL_NAME}"
policy_base_url: "${POLICY_BASE_URL}"
policy_api_key: "dummy_key"  # Not needed for local endpoints

# Connection limits (adjust based on your cluster)
global_aiohttp_connector_limit_per_host: 16384
global_aiohttp_connector_limit: 65536
```

### 2. Initialize Gym Integration

```python
from pathlib import Path
from omegaconf import DictConfig
from penguin.cli import GlobalConfigDictParserConfig, RunHelper
from penguin.rollout_collection import RolloutCollectionHelper
from penguin.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig


class GymIntegration:
    """Wrapper for Gym integration into training frameworks."""
    
    def __init__(
        self,
        model_name: str,
        base_urls: list[str],
        config_path: Path,
        head_server_host: str = "0.0.0.0",
        head_server_port: int = 8080,
    ):
        self.model_name = model_name
        self.base_urls = base_urls
        
        # Build initial config
        initial_config = {
            "policy_model_name": model_name,
            "policy_base_url": base_urls,
            "policy_api_key": "dummy_key",
            HEAD_SERVER_KEY_NAME: {
                "host": head_server_host,
                "port": head_server_port,
            },
        }
        
        # Initialize Gym
        self.run_helper = RunHelper()
        self.run_helper.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=config_path,
                initial_global_config_dict=DictConfig(initial_config),
                skip_load_from_cli=True,
            )
        )
        
        # Setup rollout collection
        self.head_server_config = BaseServerConfig(
            host=head_server_host,
            port=head_server_port,
        )
        self.rollout_helper = RolloutCollectionHelper()
    
    def shutdown(self):
        """Clean up Gym resources."""
        self.run_helper.shutdown()
```

### 3. Collect Rollouts

```python
import asyncio
from typing import Iterator


class GymIntegration:
    # ... __init__ and shutdown from above ...
    
    async def collect_rollouts(
        self,
        examples: list[dict],
    ) -> Iterator[dict]:
        """Collect rollouts for a batch of examples."""
        result_iterator = self.rollout_helper.run_examples(
            examples=examples,
            head_server_config=self.head_server_config,
        )
        
        for task in result_iterator:
            result = await task
            yield result
```

---

## Data Format

### Input Format

Convert your prompts to Gym's expected format:

```python
def prepare_gym_example(prompt: str, tools: list[dict] | None = None) -> dict:
    """Convert a prompt to Gym's expected format."""
    example = {
        "responses_create_params": {
            "input": [
                {"type": "message", "role": "user", "content": prompt}
            ],
            "model": "policy",  # Gym routes to your policy
        },
        "agent_ref": "your_agent_name",
    }
    
    if tools:
        example["responses_create_params"]["tools"] = tools
    
    return example
```

:::{tip}
Use `ng_prepare_data` to prepare datasets with proper `agent_ref` routing. Refer to {doc}`/tutorials/integrate-training-frameworks/train-with-nemo-rl` for details.
:::

### Output Format

Gym returns rollouts with token-level training data:

```{list-table} Rollout Response Fields
:header-rows: 1
:widths: 30 70

* - Field
  - Description
* - `generation_token_ids`
  - Token IDs to train on
* - `generation_log_probs`
  - Log probabilities for policy gradient
* - `prompt_token_ids`
  - Context tokens for sequence reconstruction
* - `reward`
  - Verification score (0.0–1.0)
```

:::{dropdown} Full Response Structure
:icon: code

```python
rollout = {
    "response": {
        "output": [
            {
                "type": "message",
                "role": "assistant", 
                "content": "The answer is 4.",
                "prompt_token_ids": [1, 2, 3, ...],
                "generation_token_ids": [10, 11, 12, ...],
                "generation_log_probs": [-0.5, -0.3, -0.1, ...],
            }
        ],
    },
    "reward": 1.0,
    "responses_create_params": {...},  # Original request
}
```

:::

---

## Usage

### Training Loop Integration

Add rollout collection to your existing training loop:

```python
async def training_step(
    gym: GymIntegration,
    batch: list[dict],
    policy,
    optimizer,
):
    """Single training step with Gym rollouts."""
    
    # 1. Collect rollouts through Gym
    rollouts = []
    async for rollout in gym.collect_rollouts(batch):
        rollouts.append(rollout)
    
    # 2. Process rollouts into training format
    training_data = process_rollouts(rollouts, tokenizer)
    
    # 3. Your existing training step
    loss = policy.compute_loss(training_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

:::{dropdown} Full Integration Example
:icon: code

```python
import asyncio
from pathlib import Path


async def main():
    # 1. Initialize Gym
    gym = GymIntegration(
        model_name="qwen3-4b-instruct",
        base_urls=["http://localhost:8000/v1"],
        config_path=Path("gym_env.yaml"),
    )
    
    try:
        # 2. Prepare batch
        prompts = load_training_prompts()  # Your data loading
        batch = [prepare_gym_example(p) for p in prompts[:8]]
        
        # 3. Collect rollouts
        rollouts = []
        async for rollout in gym.collect_rollouts(batch):
            rollouts.append(rollout)
            print(f"Collected rollout with reward: {rollout['reward']}")
        
        # 4. Process for training
        print(f"Collected {len(rollouts)} rollouts")
        
    finally:
        gym.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

:::

---

## Verify the Integration

Run your script and check for successful rollout collection:

```bash
python your_training_script.py
```

**Expected output**:

```text
Collected rollout with reward: 1.0
Collected rollout with reward: 0.5
Collected rollout with reward: 1.0
...
Collected 8 rollouts
```

**Success criteria**:

- Gym servers start (resource servers, agents)
- Rollouts return with rewards
- No connection errors or timeouts

---

## Troubleshooting

:::{dropdown} Connection Refused to Policy Endpoint
:icon: alert

**Symptom**: `aiohttp.ClientConnectorError: Cannot connect to host`

**Solutions**:
- Ensure your vLLM HTTP server is running
- Verify `base_urls` match your endpoint address
- Check firewall rules between training and inference nodes

:::

:::{dropdown} Agent Not Found
:icon: alert

**Symptom**: `KeyError: 'your_agent_name'`

**Solutions**:
- Ensure your Gym config includes the agent definition
- Use `ng_prepare_data` to set `agent_ref` correctly
- Verify the agent name matches your config

:::

:::{dropdown} Slow Rollout Collection
:icon: alert

**Symptom**: Rollouts take much longer than expected

**Solutions**:
- Increase `global_aiohttp_connector_limit_per_host` in your config
- Default may be too low for high parallelism
- Check network bandwidth between nodes

:::

---

## Next Step

You are collecting rollouts. Next, validate that your integration works correctly end-to-end.

:::{button-ref} validate-integration
:color: primary
:outline:

Validate Your Integration →
:::

## Resources

- {py:class}`nemo_gym.rollout_collection.RolloutCollectionHelper`
- {doc}`/about/concepts/training-integration-architecture`
