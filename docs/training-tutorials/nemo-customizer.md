(training-nemo-customizer)=
# NeMo Customizer

```{note}
This page is a stub. Content is being developed.

**Blocker**: External service dependency. See [GitHub Issue #550](https://github.com/NVIDIA-NeMo/Gym/issues/550) for coordination with NeMo Customizer team.
```

Train models with NeMo Gym using [NeMo Customizer](https://docs.nvidia.com/nemo/nemo-microservices/latest/), NVIDIA's managed fine-tuning microservice.

:::{card}

**Goal**: Fine-tune models using NeMo Customizer's API with NeMo Gym rollouts.

^^^

**In this tutorial, you will**:

1. Generate training data using NeMo Gym rollout collection
2. Configure NeMo Customizer for RL fine-tuning
3. Launch and monitor training via the Customizer API
4. Export and deploy the fine-tuned model

:::

---

## Before You Begin

- ✅ **Access**: NeMo Customizer API credentials
- ✅ **Software**: Python 3.10+, NeMo Gym installed
- ✅ **Data**: Training dataset prepared (or use NeMo Gym rollout collection)

```bash
pip install nemo-gym requests
```

## Why NeMo Customizer + NeMo Gym?

**NeMo Customizer** provides enterprise-grade fine-tuning:
- Managed infrastructure—no GPU cluster setup required
- NVIDIA-optimized training recipes for maximum efficiency
- API-driven workflow for automation and CI/CD integration
- Built-in checkpointing, monitoring, and model versioning

**NeMo Gym** complements Customizer by providing:
- Rollout collection for generating high-quality training data
- Verifiers for computing accurate task-specific rewards
- Diverse training environments (math, code, tool calling, and more)

## Getting Started

<!-- TODO: Add quick start command or notebook link -->

```python
import requests
from nemo_gym import RolloutCollectionHelper

# Step 1: Collect rollouts with NeMo Gym
# TODO: Add rollout collection code

# Step 2: Submit to NeMo Customizer API
# TODO: Add API submission code
```

## Integration Pattern

### Rollout Collection → Training Data

Use NeMo Gym to collect and verify rollouts, then format them for Customizer:

<!-- TODO: Document rollout collection to Customizer data format -->

```python
# Example: Collecting rollouts and formatting for Customizer
# TODO: Add integration code
```

### API Configuration

<!-- TODO: Document Customizer API configuration -->

```python
# NeMo Customizer API configuration
customizer_config = {
    "api_endpoint": "https://your-customizer-endpoint",
    "model": "nemotron-nano",
    # TODO: Add configuration details
}
```

## Training Workflow

### 1. Prepare Training Data

<!-- TODO: Document data preparation from Gym rollouts -->

### 2. Launch Training Job

<!-- TODO: Document API call to launch training -->

### 3. Monitor Progress

<!-- TODO: Document monitoring endpoints -->

### 4. Export Model

<!-- TODO: Document model export process -->

## Supported Features

| Feature | Status | Description |
|---------|--------|-------------|
| SFT | TBD | Supervised fine-tuning from rollouts |
| LoRA | TBD | Parameter-efficient fine-tuning |
| Full fine-tuning | TBD | Full model weight updates |

## Complete Example

<!-- TODO: Add end-to-end working example -->

## Troubleshooting

<!-- TODO: Add common issues and solutions -->

---

## What's Next?

After completing this tutorial, explore these options:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use Other Training Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse available resource servers on GitHub to find other training environments.
+++
{bdg-secondary}`github` {bdg-secondary}`resource-servers`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` NeMo RL with GRPO
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc

Multi-step tool calling with GRPO using NeMo RL.
+++
{bdg-secondary}`rl` {bdg-secondary}`grpo`
:::

::::
