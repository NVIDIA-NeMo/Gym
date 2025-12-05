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
* And more...

**NeMo Gym's Role**: Within this ecosystem, Gym focuses on standardizing scalable rollout collection for RL training. It provides unified interfaces to heterogeneous RL environments and curated resource servers with verification logic. This makes it practical to generate large-scale, high-quality training data for NeMo RL and other training frameworks.

---

## Training Framework Integrations

Gym integrates with training frameworks through OpenAI-compatible HTTP endpoints. The following frameworks have existing integrations or compatible generation backends:

### First-Party Integration

```{list-table}
:header-rows: 1
:widths: 25 25 50

* - Framework
  - Generation Backend
  - Integration Status
* - [NeMo RL](https://github.com/NVIDIA-NeMo/RL)
  - vLLM
  - ✅ Native integration via `penguin.py`
```

### Community Frameworks with Compatible Backends

These frameworks use generation backends that can expose OpenAI-compatible endpoints:

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Framework
  - Generation Backend(s)
  - Notes
* - [VeRL](https://github.com/volcengine/verl)
  - HF rollout, vLLM, SGLang
  - Multiple backend options
* - [TRL](https://github.com/huggingface/trl)
  - vLLM, HF native
  - Popular Hugging Face library
* - [Slime](https://github.com/THUDM/slime)
  - SGLang
  - Research framework
* - [OpenPipe ART](https://github.com/OpenPipe/ART)
  - vLLM
  - Async training framework
```

### Implementing Custom Integrations

If your training framework uses vLLM or SGLang but doesn't expose an OpenAI-compatible endpoint, refer to:

- {doc}`/tutorials/training-framework-integration/index` — Step-by-step integration tutorial
- {doc}`concepts/training-integration-architecture` — Architecture and design rationale

For frameworks using other generation backends, significant refactoring may be required. Refer to the architecture documentation for guidance on what Gym requires.

---

## OpenAI-Compatible Server Requirements

Gym requires generation backends to expose these endpoints:

| Endpoint | Required | Purpose |
|----------|----------|---------|
| `/v1/chat/completions` | Yes | Chat completions with tool calling |
| `/v1/models` | Yes | Model listing for validation |
| `/tokenize` | Optional | Direct tokenization access |

**Reference implementations**:

- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [SGLang OpenAI-compatible APIs](https://docs.sglang.ai/backend/openai_api_vision.html)
- [NeMo RL vLLM HTTP server](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/generation/vllm/vllm_worker_async.py)
