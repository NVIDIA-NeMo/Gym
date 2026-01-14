(about-ecosystem)=
# NeMo Gym in the Ecosystem

NeMo Gym is designed to integrate seamlessly with the broader reinforcement learning ecosystem—not just NVIDIA tools, but the entire landscape of RL training frameworks and environment libraries. This page describes how NeMo Gym fits into both the NVIDIA NeMo Framework and the wider RL community.

:::{tip}
For details on NeMo Gym capabilities, refer to the
{ref}`Overview <about-overview>`.
:::

---

## Training Framework Integrations

NeMo Gym provides scalable rollout collection infrastructure that decouples environment development from training. This design enables integration with various RL training frameworks. Below are the currently supported and planned integrations.

### Supported Frameworks

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Framework
  - Status
  - Description
* - [NeMo RL](https://github.com/NVIDIA-NeMo/RL)
  - ✅ Supported
  - NVIDIA's scalable post-training library. Supports GRPO, DPO, SFT, and on-policy distillation with PyTorch (DTensor) and Megatron Core backends. NeMo Gym integrates natively for multi-turn rollout collection and verified reward training.
* - [Unsloth](https://github.com/unslothai/unsloth)
  - ✅ Supported
  - Fast fine-tuning framework with 2-5x speedup and 80% memory reduction. Supports LoRA and QLoRA for efficient model adaptation.
```

### In-Progress Integrations

We're actively working on integrations with additional training frameworks:

| Framework | Description | Status |
|-----------|-------------|--------|
| [veRL](https://github.com/volcengine/verl) | Volcano Engine's scalable RL library with similar architectural principles | 🔜 In Progress |
| [TRL](https://github.com/huggingface/trl) | Hugging Face's Transformer Reinforcement Learning library | 🔜 In Progress |

:::{note}
**Want to integrate your training framework?** NeMo Gym outputs standardized rollout data (JSONL with OpenAI-compatible message format and verification rewards). See the {doc}`Contributing Guide <../contribute/index>` or [open an issue](https://github.com/NVIDIA-NeMo/Gym/issues) to discuss integration requirements.
:::

---

## Environment Library Integrations

NeMo Gym integrates with environment libraries to provide diverse training scenarios—from reasoning tasks to web browsing agents.

### Supported Libraries

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Library
  - Status
  - Description
* - [reasoning-gym](https://github.com/open-thought/reasoning-gym)
  - ✅ Supported
  - Procedurally generated reasoning tasks for training and evaluation. See `resources_servers/reasoning_gym/` for the integration pattern.
* - [Aviary](https://github.com/Future-House/aviary)
  - ✅ Supported
  - Multi-environment framework for tool-using agents. Supports GSM8K, HotPotQA, BixBench, and more. See `resources_servers/aviary/` for examples.
```

### In-Progress Integrations

| Library | Description | Status |
|---------|-------------|--------|
| [PRIME Intellect](https://github.com/PrimeIntellect-ai) | Distributed AI training environments | 🔜 In Progress |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | Web browsing and automation environments for agent training | 🔜 In Progress |

### Native Environment Pattern

NeMo Gym's **Resource Server** pattern provides a native way to build LLM training environments:

- **Tool definitions** with OpenAI function calling schema
- **Verification logic** for computing rewards from rollout outcomes
- **State management** for multi-step and multi-turn scenarios
- **Curated datasets** for training and evaluation

This pattern enables building environments specifically designed for LLM capabilities like tool use, instruction following, and complex reasoning—scenarios that traditional RL environments weren't designed for.

:::{seealso}
Learn how to build custom environments in the {doc}`Creating a Resource Server <../tutorials/creating-resource-server>` tutorial.
:::

---

## NeMo Gym Within the NeMo Framework

NeMo Gym is a component of the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/), NVIDIA's GPU-accelerated platform for building and training generative AI models.

NeMo Framework includes modular libraries for end-to-end model development:

| Library | Purpose |
|---------|---------|
| [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | Pretraining and fine-tuning with Megatron-Core |
| [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) | PyTorch native training for Hugging Face models |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | Scalable and efficient post-training |
| **[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)** | RL environment infrastructure and rollout collection *(this project)* |
| [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | Data preprocessing and curation |
| [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) | Synthetic data generation from scratch or seed datasets |
| [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | Model evaluation and benchmarking |
| [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Programmable safety guardrails |

**NeMo Gym's Role**: Within this ecosystem, Gym focuses on standardizing scalable rollout collection for RL training. It provides unified interfaces to heterogeneous RL environments and curated resource servers with verification logic. This makes it practical to generate large-scale, high-quality training data for NeMo RL and other training frameworks.

---

## Community & Contributions

NeMo Gym thrives on community contributions. Here's how you can get involved:

- **Add a training framework integration**: Help connect NeMo Gym with your favorite RL training library
- **Contribute a resource server**: Share environments for domains like coding, math, tool use, or instruction following
- **Improve documentation**: Help others get started with clearer guides and examples
- **Report issues and suggest features**: Your feedback shapes NeMo Gym's roadmap

:::{tip}
**Getting started**: Check out the {doc}`Contributing Guide <../contribute/index>` or browse [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) to find areas where you can help.
:::
