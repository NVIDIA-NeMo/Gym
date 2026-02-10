(about-ecosystem)=
# NeMo Gym in the Ecosystem

We're building NeMo Gym to integrate with a broad set of RL training frameworks and environment libraries.

We would love your contribution! Open a PR to add an integration, or [file an issue](https://github.com/NVIDIA-NeMo/Gym/issues/new/choose) to share what would be valuable for you.

---

## Supported Training Framework Integrations

NeMo Gym decouples environment development from training by outputting standardized JSONL rollout data. Training frameworks consume this data through their own integration code.

- **[NeMo RL](https://github.com/NVIDIA/NeMo-RL)** — {doc}`Tutorial <../tutorials/nemo-rl-grpo/index>`
- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)** — [Integration](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_nemogym_executor.py)
- **[TRL](https://github.com/huggingface/trl)** — {doc}`Tutorial <../training-tutorials/trl>`
- **[Unsloth](https://github.com/unslothai/unsloth)** — {doc}`Tutorial <../tutorials/unsloth-training>`
- **NeMo Customizer** — *(In progress)*
- **[veRL](https://github.com/volcengine/verl)** — *(In progress)*



:::{tip}
**Integrate your framework**: Refer to the {doc}`Training Framework Integration Guide <../contribute/rl-framework-integration/index>` or [open an issue](https://github.com/NVIDIA-NeMo/Gym/issues) to discuss requirements.
:::

---

## Environment Library Integrations

NeMo Gym integrates with environment libraries to provide diverse training scenarios, from reasoning tasks to tool-using agents.

### Supported Libraries

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Library
  - Status
  - Description
* - [reasoning-gym](https://github.com/open-thought/reasoning-gym)
  - ✅ Code
  - Procedurally generated reasoning tasks. Integration: [`resources_servers/reasoning_gym/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/reasoning_gym)
* - [Aviary](https://github.com/Future-House/aviary)
  - ✅ Code
  - Multi-environment framework for tool-using agents (GSM8K, HotPotQA, BixBench). Integration: [`resources_servers/aviary/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/aviary)
```

### In-Progress Integrations

| Library | Description | Status |
|---------|-------------|--------|
| [PRIME Intellect](https://github.com/PrimeIntellect-ai) | Distributed AI training environments | 🔜 Planned |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | Web browsing and automation environments | 🔜 Planned |

### Building Custom Environments

Beyond external library integrations, NeMo Gym provides a native pattern for building LLM training environments—the {term}`Resource Server <Verifier>`. This pattern has four components:

- **Tool definitions**: OpenAI function calling schema for model interactions
- **Verification logic**: Computes reward scores (0.0-1.0) from rollout outcomes
- **State management**: Tracks context across {term}`multi-step` and {term}`multi-turn` interactions
- **Curated datasets**: Task prompts paired with expected outcomes

This pattern supports LLM-specific capabilities like tool use, instruction following, and complex reasoning that traditional RL environments were not designed for.

:::{seealso}
Refer to the {doc}`Creating a Resource Server <../tutorials/creating-resource-server>` tutorial to build custom environments.
:::

---

## NeMo Gym Within the NeMo Framework

NeMo Gym is a component of the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/), a GPU-accelerated platform for building and training generative AI models.

The NeMo Framework includes modular libraries for end-to-end model development:

| Library | Purpose |
|---------|---------|
| [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | Pretraining and fine-tuning with Megatron-Core |
| [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) | PyTorch native training for Hugging Face models |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | Scalable post-training with GRPO, DPO, and SFT |
| **[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)** | RL environment infrastructure and rollout collection *(this project)* |
| [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | Data preprocessing and curation |
| [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) | Synthetic data generation |
| [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | Model evaluation and benchmarking |
| [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Programmable safety guardrails |

**NeMo Gym's role**: Gym standardizes rollout collection for RL training. It provides unified interfaces to heterogeneous RL environments and resource servers with verification logic, enabling large-scale training data generation for NeMo RL and other frameworks.

---

## Community and Contributions

NeMo Gym welcomes community contributions in the following areas:

- **Training framework integration**: Connect NeMo Gym with additional RL training libraries
- **Resource server contributions**: Share environments for domains like coding, math, tool use, or instruction following
- **Documentation improvements**: Improve guides and examples for new users
- **Issue reporting**: Report bugs and suggest features to shape the roadmap

:::{tip}
Refer to the {doc}`Contributing Guide <../contribute/index>` or browse [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) to get started.
:::
