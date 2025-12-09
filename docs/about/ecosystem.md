(about-ecosystem)=
# NeMo Gym in the NVIDIA Ecosystem

NeMo Gym is a component of the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/), NVIDIA's GPU-accelerated platform for building and training generative AI models.

:::{tip}
For details on NeMo Gym capabilities, refer to the
{ref}`Overview <about-overview>`.
:::

---

## NeMo Gym Within the NeMo Framework

The [NeMo Framework](https://github.com/NVIDIA-NeMo) is NVIDIA's GPU-accelerated platform for training large language models (LLMs), multimodal models, and speech models. It includes multiple specialized components:

* **NeMo Megatron-Bridge**: Pretraining and fine-tuning with Megatron-Core
* **NeMo AutoModel**: PyTorch native training for Hugging Face models
* **NeMo RL**: Scalable reinforcement learning toolkit
* **NeMo Gym**: RL environment infrastructure and rollout collection (this project)
* **NeMo Curator**: Data preprocessing and curation
* **NeMo Data Designer**: Synthetic data generation for post-training
* **NeMo Evaluator**: Model evaluation and benchmarking
* **NeMo Guardrails**: Programmable safety guardrails
* And more...

**NeMo Gym's Role**: Within this ecosystem, Gym focuses on standardizing scalable rollout collection for RL training. It provides unified interfaces to heterogeneous RL environments and curated resource servers with verification logic. This makes it practical to generate large-scale, high-quality training data for NeMo RL and other training frameworks.

---

## NeMo Gym and NeMo Data Designer

[NeMo Data Designer](https://nvidia-nemo.github.io/DataDesigner/) is a general framework for generating high-quality synthetic data from scratch or using seed data. Both tools generate training data, but they serve different use cases and employ different generation strategies.

### Key Differences

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} NeMo Data Designer
**Synthetic data generation**

Generates training data using LLM prompting combined with statistical samplers. Data Designer excels at creating diverse datasets with controlled distributions, meaningful correlations between fields, and built-in validation.

**Best for:**
- Generating diverse post-training datasets at scale
- Creating data with specific statistical properties
- Simulating tool call patterns and responses
- Rapid iteration on data characteristics
:::

:::{grid-item-card} NeMo Gym
**Real environment interactions**

Generates training data through actual interactions with live environments. Gym executes real tool calls, runs actual verification logic, and produces reward scores from genuine environment feedback.

**Best for:**
- Collecting rollouts with verified reward signals
- Training agents that need real tool execution
- Environments requiring actual API calls or code execution
- RL training with ground-truth verification
:::

::::

### When to Use Each Tool

| Use Case | Recommended Tool |
|----------|------------------|
| Generate diverse SFT data with controlled distributions | Data Designer |
| Simulate tool calling patterns for initial training | Data Designer |
| Collect RL rollouts with real reward signals | Gym |
| Execute actual tools (code, APIs, search) during generation | Gym |
| Create datasets with statistical diversity guarantees | Data Designer |
| Train agents on verified real-world interactions | Gym |
| Rapid prototyping of data characteristics | Data Designer |
| Ground-truth verification from live environments | Gym |

### Complementary Workflows

Data Designer and Gym complement each other in a typical training pipeline:

1. **Bootstrap with Data Designer**: Generate initial synthetic datasets for supervised fine-tuning (SFT). Use statistical samplers to ensure diversity and LLM columns to create realistic tool-calling patterns.

2. **Refine with Gym**: Transition to Gym for reinforcement learning. Collect rollouts from real environment interactions where tool calls execute against actual systems and verification produces ground-truth rewards.

3. **Iterate**: Use insights from Gym rollouts to refine Data Designer configurations for the next training cycle.

:::{seealso}
- [NeMo Data Designer Documentation](https://nvidia-nemo.github.io/DataDesigner/)
- [NeMo Data Designer GitHub Repository](https://github.com/NVIDIA-NeMo/DataDesigner)
:::
