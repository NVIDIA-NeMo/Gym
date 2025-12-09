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
* **NeMo Gym**: RL training environment development (this project)
* **NeMo Curator**: Data preprocessing and curation
* **NeMo Data Designer**: Synthetic data generation
* **NeMo Evaluator**: Model evaluation and benchmarking
* **NeMo Guardrails**: Programmable safety guardrails
* And more...

**NeMo Gym's Role**: Within this ecosystem, Gym focuses on improving developer experience during RL training environment development and usage. Typically Gym training environments will be used with NeMo RL to perform RL training.

---

## NeMo Gym and NeMo Data Designer

[NeMo Data Designer](https://nvidia-nemo.github.io/DataDesigner/) is a general framework for synthetic data generation from scratch or using seed data. It helps you generate high quality and diversity data.

Gym is a training environment framework with an emphasis on training environment developer experience. It helps you develop a training environment and use it in downstream RL training.

:::{seealso}
- [NeMo Data Designer Documentation](https://nvidia-nemo.github.io/DataDesigner/)
- [NeMo Data Designer GitHub Repository](https://github.com/NVIDIA-NeMo/DataDesigner)
:::
