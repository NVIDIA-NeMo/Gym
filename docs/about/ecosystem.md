(about-ecosystem)=
# NeMo Gym in the NVIDIA Ecosystem

NeMo Gym is a component of the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/), NVIDIA's GPU-accelerated platform for building and training generative AI models.

:::{tip}
For details on NeMo Gym capabilities, refer to the
{ref}`Overview <about-overview>`.
:::

---

## NeMo Gym Within the NeMo Framework

NeMo Framework includes modular libraries for end-to-end model training:

* **[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)**: Pretraining and fine-tuning with Megatron-Core
* **[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)**: PyTorch native training for Hugging Face models
* **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)**: Scalable and efficient post-training
* **[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)**: RL environment infrastructure and rollout collection (this project)
* **[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)**: Data preprocessing and curation
* **[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)**: Synthetic data generation from scratch or seed datasets
* **[NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)**: Model evaluation and benchmarking
* **[NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)**: Programmable safety guardrails
* And more...

**NeMo Gym's Role**: Within this ecosystem, Gym focuses on standardizing scalable rollout collection for RL training. It provides unified interfaces to heterogeneous RL environments and curated resource servers with verification logic. This makes it practical to generate large-scale, high-quality training data for NeMo RL and other training frameworks.
