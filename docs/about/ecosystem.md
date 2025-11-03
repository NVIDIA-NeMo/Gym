(about-ecosystem)=
# NeMo Gym in the NVIDIA Ecosystem

NeMo Gym is a component of the [NVIDIA NeMo Framework](https://github.com/NVIDIA-NeMo), NVIDIA's GPU-accelerated platform for building and training generative AI models.

:::{tip}
For details on NeMo Gym capabilities, refer to the
{ref}`Overview <about-overview>` and {ref}`Key Features <about-features>`. For internal design and
boundaries, refer to {ref}`Architecture <about-architecture>`.
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

**NeMo Gym's Role**: Within this ecosystem, Gym focuses specifically on standardizing scalable rollout collection for RL training. It provides unified interfaces to heterogeneous RL environments and curated resource servers with verification logic, making it practical to generate large-scale, high-quality training data that feeds into NeMo RL and other training frameworks.

---

## Comparing NeMo Gym with NeMo Agent Toolkit

While NeMo Gym is part of the NeMo Framework for **training**, the [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) is a separate open-source framework focused on **inference** and building production agent workflows.

::::{tab-set}

:::{tab-item} NeMo Gym
**Focus**: Training

Part of the NeMo Framework. Provides RL environments and rollout collection for model training.

**Best for**: Teams training models with RL and needing high-throughput data collection with verification.

**Key characteristics**:
* Unified interfaces to heterogeneous RL environments
* Passes trajectory data to RL training frameworks
* Curated environments with verification logic for reward signals
* Parallel and asynchronous rollout collection for scale
:::

:::{tab-item} NeMo Agent Toolkit
**Focus**: Inference

An open-source framework that is interoperable with other ecosystems and supports end-to-end profiling and optimization of complex agentic systems.

**Best for**: Teams building and operating agent workflows for inference, needing observability, profiling, and evaluation.

**Key characteristics**:
* Interoperable with other frameworks
* End-to-end profiling and optimization for agent pipelines
* Helps expose bottlenecks and costs to scale agent systems reliably
:::

::::

---

(ecosystem-comparison)=
## Choosing Between NeMo Gym and NeMo Agent Toolkit

The right choice depends on whether you are training models with RL (use NeMo Gym within NeMo Framework) or building production agent workflows (use NeMo Agent Toolkit).

### Feature Comparison

```{list-table}
::header-rows: 1
::widths: 28 28 28 16

* - Aspect
  - NeMo Gym
  - NeMo Agent Toolkit
  - Both (Integrated)
* - **Target Users**
  - ML engineers focused on RL training and data generation
  - Developers and platform engineers building agentic workflows
  - Teams connecting agent workflows to RL training
* - **Primary Use Case**
  - Scalable rollout collection with verification for RL
  - Build, observe, and optimize agent systems for inference
  - Use agent trajectories as training data for RL
* - **Trajectory Collection**
  - Collects at scale with parallel/async execution
  - Collects trajectories from running agent workflows
  - NAT collects → Gym consumes for RL
* - **Evaluation**
  - Verification logic to derive reward signals for RL
  - Full evaluation harnesses for agent workflows
  - Use NAT eval to score, use Gym for RL rewards
* - **Profiling / Instrumentation**
  - Training-focused metrics for RL efficiency
  - Inference-focused metrics (per-invocation, cost, sizing)
  - Complementary views across training and inference
* - **Observability**
  - RL training metrics (e.g., mixed-reward, truncation rate)
  - Telemetry exporters and observability platform integration
  - End-to-end visibility across the pipeline
* - **Agent Orchestration**
  - Minimal agent for-loop to unblock RL research/training
  - Multiple agent implementations with abstractions and interfaces
  - Use NAT orchestration, send trajectories to Gym
* - **LLM Integration**
  - Via RL frameworks that handle training and generation
  - Interfaces to inference-only LLM endpoints
  - Bridge training and inference paths
* - **Framework Connections**
  - Passes trajectory data to RL frameworks (VeRL, NeMo-RL, OpenRLHF)
  - Agent frameworks (e.g., LangChain, LlamaIndex, LangGraph)
  - Connect agent and RL frameworks through shared data
* - **UI**
  - None
  - Available
  - Use NAT UI while training with Gym
* - **MCP (Model Context Protocol)**
  - Planned: MCP client support for training on external tools
  - Acts as client or server to use or publish tools
  - Future: Combine to train on MCP-powered tool use
```

### Decision Guide

```{list-table}
:header-rows: 1
:widths: 30 70

* - Scenario
  - When to Use
* - **Use NeMo Gym**
  - • You want to train a model with your preferred RL framework <br>
    • You need scalable rollout collection with verification for reward signals <br>
    • You prefer minimal agent orchestration tailored for RL research and training
* - **Use NeMo Agent Toolkit**
  - • You are building agent workflows with observability, profiling, and evaluation <br>
    • You need inference-first instrumentation and cost visibility <br>
    • You want to work within an open and interoperable agent framework
* - **Use Both (NAT + Gym)**
  - • You want to train models using data from agent workflows <br>
    • Your agent framework (NAT or another) collects trajectories and scores them, then Gym consumes that data for RL with frameworks such as VeRL or NeMo RL
```

---

## Integration Points

NeMo Gym and NeMo Agent Toolkit can be composed in several ways to support RL training using agent-generated data. Choose the integration pattern that matches your setup:

::::{tab-set}

:::{tab-item} NAT → Gym
**Pattern**: NAT Workflow as an Endpoint → Gym for RL

Use this when you have NeMo Agent Toolkit workflows and want to use them for RL training.

**Workflow**:
1. Serve a NAT agent workflow as an endpoint
2. Gym triggers runs and requests trajectory collection
3. NAT executes the workflow and collects trajectories
4. NAT evaluates and assigns scores/rewards
5. Gym forwards trajectories and rewards to the chosen RL framework for backpropagation and checkpoint updates

**Best for**: Teams already using NAT who want to leverage agent workflows for model training
:::

:::{tab-item} Other Frameworks → Gym
**Pattern**: Agent Frameworks → Gym for RL

Use this when you have existing agent workflows in LangChain, LangGraph, CrewAI, or similar frameworks.

**Workflow**:
1. Serve the existing agent workflow as an endpoint
2. Gym triggers runs and requests trajectory collection
3. The workflow collects trajectories and performs evaluation
4. Scores and trajectories are passed to Gym
5. Gym hands data to the RL framework for training

**Best for**: Teams with established agent frameworks who want to add RL training capabilities
:::

:::{tab-item} Gym Only
**Pattern**: Model-Only Training Inside Gym

Use this when you want to train models directly against resource servers without external agent frameworks.

**Workflow**:
1. Run the model within Gym against resource servers (tools and verification)
2. Gym captures trajectories produced during interactions
3. Gym sends trajectories to the RL framework for training

**Best for**: Teams focused on RL research and training without complex agent orchestration requirements
:::

::::

---

## Related NVIDIA Components

* **[NeMo Framework](https://github.com/NVIDIA-NeMo)**: The parent ecosystem containing NeMo Gym, NeMo RL, NeMo Curator, and other training components
* **[NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit)**: Separate framework for building production agent workflows with observability and profiling
* **NeMo-RL**: Training framework within NeMo Framework that consumes trajectory data from NeMo Gym
* **VeRL**, **OpenRLHF**: External RL training frameworks compatible with NeMo Gym data formats
* **NVIDIA NIM**: Inference microservices that can be paired with agent workflows and training pipelines

