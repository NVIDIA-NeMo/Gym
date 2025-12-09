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
* **NeMo Evaluator**: Model evaluation and benchmarking
* **NeMo Guardrails**: Programmable safety guardrails
* **NeMo Data Designer**: Synthetic data generation from scratch or seed datasets

**NeMo Gym's Role**: Within this ecosystem, Gym focuses on standardizing scalable rollout collection for RL training. It provides unified interfaces to heterogeneous RL environments and curated resource servers with verification logic. This makes it practical to generate large-scale, high-quality training data for NeMo RL and other training frameworks.
