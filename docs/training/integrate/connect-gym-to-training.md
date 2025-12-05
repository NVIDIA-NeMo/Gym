(integrate-connect-gym-to-training)=

# Connect Gym to Your Training Loop

Integrate Gym's rollout collection into your custom RL training pipeline.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
30 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`expose-openai-endpoint`
- Working OpenAI-compatible endpoint
- Existing training loop that needs rollouts

:::

::::

---

## Goal

By the end of this guide, your training loop will:

1. Initialize Gym's rollout collection system
2. Send batches of prompts to Gym
3. Receive rollouts with token IDs, log probabilities, and rewards
4. Feed results into your training step

---

## Install Gym in Your Training Environment

Add Gym as a dependency in your training environment:

```bash
# From your training framework directory
pip install nemo-gym
```

Or add to your `pyproject.toml`:

```toml
[project.optional-dependencies]
gym = ["nemo-gym>=0.1.0"]
```

---

## Create a Gym Configuration File

Gym needs to know about your resource servers and agents. Create `gym_env.yaml` in your training directory:

<!-- TODO: SME to verify minimal config structure -->

```yaml
# gym_env.yaml - Minimal config for training integration

# Policy model connection (filled at runtime)
policy_model_name: "${POLICY_MODEL_NAME}"
policy_base_url: "${POLICY_BASE_URL}"
policy_api_key: "dummy_key"  # Not needed for local endpoints

# Connection limits (adjust based on your cluster)
global_aiohttp_connector_limit_per_host: 16384
global_aiohttp_connector_limit: 65536
```

---

## Initialize the Gym RunHelper

In your training script, initialize Gym before your training loop:

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
        self.head_server_host = head_server_host
        self.head_server_port = head_server_port
        
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

---

## Integrate with Your Training Loop

Add rollout collection to your existing training loop:

```python
import asyncio
from typing import Iterator


class GymIntegration:
    # ... __init__ and shutdown from above ...
    
    async def collect_rollouts(
        self,
        examples: list[dict],
    ) -> Iterator[dict]:
        """
        Collect rollouts for a batch of examples.
        
        Args:
            examples: List of Gym-formatted examples with 'responses_create_params'
            
        Yields:
            Rollout results with token_ids, log_probs, and rewards
        """
        result_iterator = self.rollout_helper.run_examples(
            examples=examples,
            head_server_config=self.head_server_config,
        )
        
        for task in result_iterator:
            result = await task
            yield result


# In your training script:
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
    
    # 2. Process rollouts into training format (see next guide)
    training_data = process_rollouts(rollouts, tokenizer)
    
    # 3. Your existing training step
    loss = policy.compute_loss(training_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## Prepare Your Data Format

Gym expects examples in a specific format. Convert your prompts:

```python
def prepare_gym_example(prompt: str, tools: list[dict] | None = None) -> dict:
    """
    Convert a prompt to Gym's expected format.
    
    Args:
        prompt: The user prompt text
        tools: Optional list of tool definitions
        
    Returns:
        Gym-formatted example dict
    """
    example = {
        "responses_create_params": {
            "input": [
                {"type": "message", "role": "user", "content": prompt}
            ],
            "model": "policy",  # Gym routes to your policy
        },
        # Agent routing (set by ng_prepare_data or manually)
        "agent_ref": "your_agent_name",
    }
    
    if tools:
        example["responses_create_params"]["tools"] = tools
    
    return example


# Prepare a batch
prompts = ["What is 2+2?", "Explain quantum computing"]
batch = [prepare_gym_example(p) for p in prompts]
```

:::{tip}
Use `ng_prepare_data` to prepare datasets with proper `agent_ref` routing. Refer to {doc}`/tutorials/integrate-training-frameworks/train-with-nemo-rl` for details.
:::

---

## Handle the Rollout Response

Gym returns rich rollout data. Here's the structure:

```python
# Example rollout result structure
rollout = {
    "response": {
        "output": [
            {
                "type": "message",
                "role": "assistant", 
                "content": "The answer is 4.",
                # Token-level data for training
                "prompt_token_ids": [1, 2, 3, ...],
                "generation_token_ids": [10, 11, 12, ...],
                "generation_log_probs": [-0.5, -0.3, -0.1, ...],
            }
        ],
    },
    # Reward from verification
    "reward": 1.0,
    
    # Original request (for reference)
    "responses_create_params": {...},
}
```

**Key fields for training**:

| Field | Purpose |
|-------|---------|
| `generation_token_ids` | Token IDs to train on |
| `generation_log_probs` | Log probabilities for policy gradient |
| `prompt_token_ids` | Context tokens (for sequence reconstruction) |
| `reward` | Verification score (0.0-1.0) |

---

## Full Integration Example

Here's a complete example putting it all together:

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
        
        # 4. Process for training (see next guide)
        print(f"Collected {len(rollouts)} rollouts")
        
    finally:
        gym.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Verify the Integration

Run your script and check for successful rollout collection:

```bash
python your_training_script.py
```

**✅ Success**: You should see:

1. Gym servers starting (resource servers, agents)
2. Rollouts being collected with rewards
3. No connection errors or timeouts

```text
Collected rollout with reward: 1.0
Collected rollout with reward: 0.5
Collected rollout with reward: 1.0
...
Collected 8 rollouts
```

---

## Troubleshooting

### Connection refused to policy endpoint

**Symptom**: `aiohttp.ClientConnectorError: Cannot connect to host`

**Fix**: Ensure your vLLM HTTP server is running and the `base_urls` match.

### Agent not found

**Symptom**: `KeyError: 'your_agent_name'`

**Fix**: Ensure your Gym config includes the agent definition, or use `ng_prepare_data` to set `agent_ref`.

### Slow rollout collection

**Symptom**: Rollouts take much longer than expected

**Fix**: Increase `global_aiohttp_connector_limit_per_host` in your config. Default may be too low for high parallelism.

---

## Next Step

You're collecting rollouts. Next, learn how to process multi-turn rollouts into training-ready token sequences.

:::{button-ref} process-multi-turn-rollouts
:color: primary
:outline:

Next: Process Multi-Turn Rollouts →
:::

---

## Reference

- {py:class}`nemo_gym.rollout_collection.RolloutCollectionHelper`
- {doc}`/about/concepts/training-integration-architecture`

