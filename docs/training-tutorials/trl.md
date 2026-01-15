(training-trl)=

# TRL Training

```{warning}
**Status: In Development** — TRL integration is planned but not yet implemented. Track progress at [GitHub Issue #548](https://github.com/NVIDIA-NeMo/Gym/issues/548).

Looking to train now? Use {doc}`NeMo RL <../tutorials/nemo-rl-grpo/index>` (production-ready) or {doc}`Unsloth <../tutorials/unsloth-training>` (single GPU).
```

Train models using [Hugging Face TRL](https://huggingface.co/docs/trl) with NeMo Gym verifiers as reward functions.

## Why TRL + NeMo Gym?

**TRL** provides production-ready RL training for large language models:

- PPO, DPO, ORPO algorithms out of the box
- Seamless HuggingFace Hub integration
- Active community and documentation

**NeMo Gym** adds:

- Domain-specific verifiers (math, code, tool calling)
- Standardized reward computation via HTTP API
- Pre-built training environments

## Planned Integration Pattern

When implemented, TRL integration will use NeMo Gym verifiers as reward functions via HTTP:

```text
┌─────────────┐     HTTP      ┌──────────────────────┐
│  TRL Trainer │ ──────────▶ │  NeMo Gym Resource   │
│  (PPO/DPO)   │             │  Server (Verifier)   │
│              │ ◀────────── │                      │
│              │   reward    │                      │
└─────────────┘              └──────────────────────┘
```

### Proposed Reward Wrapper

The integration will expose NeMo Gym verifiers as TRL-compatible reward functions:

```python
# Proposed implementation (not yet available)
# Location: nemo_gym/integrations/trl.py

from typing import List

class NeMoGymRewardFunction:
    """Wrap a NeMo Gym resource server as a TRL reward function."""
    
    def __init__(self, resources_server_url: str):
        from nemo_gym import ResourcesServerClient
        self.client = ResourcesServerClient(resources_server_url)
    
    def __call__(self, outputs: List[str]) -> List[float]:
        """Compute rewards for a batch of model outputs."""
        rewards = []
        for output in outputs:
            response = self.client.verify(output)
            rewards.append(response.reward)
        return rewards
```

### Usage Pattern (Planned)

```python
# Planned usage (not yet available)
from trl import PPOTrainer, PPOConfig
from nemo_gym.integrations.trl import NeMoGymRewardFunction

# Start a resource server (e.g., math verification)
# nemo-gym serve math --port 8080

# Create reward function pointing to the server
reward_fn = NeMoGymRewardFunction("http://localhost:8080")

# Use with TRL trainer
config = PPOConfig(...)
trainer = PPOTrainer(config=config, reward_model=reward_fn)
trainer.train()
```

## Target Algorithms

| Algorithm | TRL Support | NeMo Gym Integration |
|-----------|-------------|----------------------|
| PPO | ✅ Stable | 🔜 Planned |
| DPO | ✅ Stable | 🔜 Planned |
| ORPO | ✅ Stable | 🔜 Planned |
| GRPO | ❌ Not in TRL | ✅ Use {doc}`NeMo RL <../tutorials/nemo-rl-grpo/index>` |

## Architecture Considerations

For architects evaluating this integration:

**Network Latency**
: Verifier calls add HTTP round-trip latency per batch. Mitigate with batched verification and local resource servers.

**Distributed Training**
: Each TRL worker connects to the same or separate resource servers. Compatibility with FSDP and DeepSpeed training modes is a design goal.

**Error Handling**
: Planned retry logic and timeout configuration for verifier failures during training.

## Contributing

Help move TRL integration forward:

1. **Track progress**: Watch [Issue #548](https://github.com/NVIDIA-NeMo/Gym/issues/548)
2. **Contribute**: See {doc}`../contribute/rl-framework-integration/index` for integration guidelines

## Available Alternatives

Ready to train today? These integrations work now:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL with GRPO
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc

Production-ready multi-node training with GRPO algorithm.
+++
{bdg-success}`available` {bdg-primary}`recommended`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth Training
:link: ../tutorials/unsloth-training
:link-type: doc

Fast, memory-efficient fine-tuning on a single GPU.
+++
{bdg-success}`available` {bdg-secondary}`single-gpu`
:::

::::
