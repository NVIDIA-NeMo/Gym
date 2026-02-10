(about-ecosystem)=
# Agentic RL Ecosystem

We're building NeMo Gym to integrate with a broad set of RL training frameworks and environment libraries.

We would love your contribution! Open a PR to add an integration, or [file an issue](https://github.com/NVIDIA-NeMo/Gym/issues/new/choose) to share what would be valuable for you.

---

## Training Framework Integrations

- **{doc}`NeMo RL <../training-tutorials/nemo-rl-grpo/index>`** - GRPO training to improve multi-step tool calling on the Workplace Assistant environment
- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_nemogym_executor.py)** - example agent executor for RL training
- **{doc}`TRL <../training-tutorials/trl>`** - GRPO training on Workplace Assistant and Reasoning Gym environments
- **{doc}`Unsloth <../training-tutorials/unsloth-training>`** - GRPO training on Sudoku environment
- **NeMo Customizer** - *(In progress)*
- **VeRL** - *(In progress)*

To integrate another training framework, see the {doc}`Training Framework Integration Guide <../contribute/rl-framework-integration/index>`.

---

## Environment Library Integrations

NeMo Gym integrates with external environment libraries and benchmarks. See the [README](https://github.com/NVIDIA-NeMo/Gym?tab=readme-ov-file#table-2-resource-servers-for-training) for the full list—here are a few examples:

- **[Reasoning Gym](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/reasoning_gym)** - reasoning environments spanning computation, cognition, logic and more
- **[Aviary](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/aviary)** - environments spanning math, knowledge, biological sequences, scientific literature search, and protein stability
- **[Verifiers](https://github.com/PrimeIntellect-ai/verifiers)** - *(In progress)* - environments spanning coding, data & ML, science & reasoning, tool use and more
- **[BrowserGym](https://github.com/ServiceNow/BrowserGym)** - *(In progress)* - environments for web task automation


---

## Related NeMo Libraries

NeMo Gym is a component of NVIDIA NeMo, a GPU-accelerated platform for building and training generative AI models.

Depending on your workflow, you may also find these libraries useful:

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