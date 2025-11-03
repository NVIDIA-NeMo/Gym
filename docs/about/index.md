---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

NeMo Gym standardizes scalable rollout collection for LLM and VLM training. It provides the infrastructure to systematically generate agent interaction data and a curated collection of high-quality RL environments, making it easy to produce large-scale training data for reinforcement learning workflows using the framework of your choice.

By offering unified interfaces to heterogeneous RL environments and seamless data handoff to popular RL training frameworks (VeRL, NeMo-RL, OpenRLHF), NeMo Gym lets you focus on research and model improvement rather than infrastructure and orchestration.

---

## The Challenge: Scaling RL Training Data

Training AI agents through reinforcement learning requires massive amounts of high-quality interaction data. Teams face several obstacles:

* **Infrastructure overhead**: Building systems to coordinate models, tools, and verification logic at scale
* **Inconsistent interfaces**: Each environment requires custom integration code and data formats
* **Quality verification**: Ensuring agent responses are accurately evaluated for reward signals
* **Framework fragmentation**: Difficulty connecting custom environments to different RL training frameworks
* **Throughput bottlenecks**: Sequential processing cannot generate data fast enough for large-scale training

These challenges slow research iteration and make it difficult to experiment with different environments and training approaches.

---

## NeMo Gym's Approach

NeMo Gym solves these problems through three core design principles:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Design Principle
  - How It Helps
* - **Unified Interfaces**
  - Standard abstractions for models (LLM inference), resources (tools + verification), and agents (orchestration) enable mix-and-match composition. Connect any model to any environment using consistent APIs based on OpenAI's Responses format.
* - **Curated Environments**
  - A growing collection of resource servers across domains (mathematics, coding, knowledge, instruction-following, agent workflows) provides both the tools agents can use and the verification logic to score their performance. Each environment includes training datasets, validation sets, and examples.
* - **High-Throughput Architecture**
  - Async orchestration with configurable parallelism supports generating thousands of rollouts concurrently. The system handles complex coordination (model inference, tool calls, verification) while maximizing throughput for training workloads.
```

---

## Core Components

NeMo Gym organizes around three core abstractions that work together:

* **Models**: LLM inference endpoints that generate text and make tool-calling decisions. Models are stateless and handle single-turn generation. Configure using OpenAI-compatible APIs or local vLLM servers.

* **Resources**: Servers that provide both tools (functions agents can call) and verifiers (logic to evaluate agent performance and assign reward signals). Examples include mathematical reasoning environments, code execution sandboxes, web search tools, and custom verification logic.

* **Agents**: Orchestration layers that connect models to resources, handle multi-turn conversations, route tool calls, and format responses consistently. Agents coordinate the interaction loop and can be extended with custom logic.

These components communicate via HTTP APIs and can run as separate services, enabling flexible deployment and scaling.

:::{seealso}
For detailed explanations of these abstractions, see {doc}`Concepts <concepts/index>`.
:::

---

## Target Users

NeMo Gym serves three primary user journeys. Choose the one that best describes your use case:

::::{tab-set}

:::{tab-item} Model Training Researchers
**Goal**: Train models using RL with my preferred training framework

**Stack**: NeMo Gym + RL Frameworks (VeRL, NeMo-RL, OpenRLHF, etc.)

You want to train a model to improve tool calling, reasoning, or task performance using reinforcement learning. NeMo Gym provides the infrastructure to generate training rollouts at scale from curated environments or your own custom verification logic, outputting data in formats compatible with your chosen RL framework.

**Key benefits**: Direct access to curated RL environments, high-throughput rollout collection, framework-agnostic output
:::

:::{tab-item} NeMo Agent Toolkit Users
**Goal**: Train models using data from my agent workflow built with NeMo Agent Toolkit

**Stack**: NAT + NeMo Gym + RL Frameworks

You have built an agent using NVIDIA NeMo Agent Toolkit (NAT) and want to improve your underlying model through RL. NAT handles trajectory collection and scoring from your agent workflow, then passes the data to NeMo Gym, which integrates with RL frameworks for model weight updates through backpropagation.

**Key benefits**: Seamless NAT integration, leverage existing agent workflows, end-to-end training pipeline
:::

:::{tab-item} Other Framework Users
**Goal**: Train models using data from my agent workflow built with LangChain, LangGraph, CrewAI, or other frameworks

**Stack**: Your Framework + NeMo Gym + RL Frameworks

Similar to the NAT journey, your existing agent framework collects trajectories and scores them via its evaluation system. NeMo Gym provides the bridge to RL training frameworks, handling data format conversion and training orchestration.

**Key benefits**: Framework-agnostic integration, reuse existing agent code, standardized RL training path
:::

::::

---

## What You Can Build

NeMo Gym enables several key use cases:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Use Case
  - Description
* - **RL Training Pipelines**
  - Generate training data at scale for reinforcement learning algorithms. Collect rollouts with reward signals, export in RL-framework-compatible formats, and feed directly into training loops.
* - **Benchmark Evaluation**
  - Systematically evaluate agent performance across tasks. Use curated datasets and verification logic to measure accuracy, reasoning quality, tool-use effectiveness, and instruction-following capabilities.
* - **Custom Environments**
  - Build your own resource servers with domain-specific tools and verification logic. NeMo Gym's base classes and templates make it straightforward to add new environments that integrate seamlessly with the rest of the system.
* - **Multi-Framework Experiments**
  - Compare different RL algorithms and training frameworks using the same environment and data. NeMo Gym collects trajectories and passes data to your choice of RL framework (VeRL, NeMo-RL, OpenRLHF) without rewriting environment code.
```

---

## Learn More

Explore the documentation to understand NeMo Gym's architecture, concepts, and capabilities:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Core Concepts
:link: concepts/index
:link-type: doc

Understand the fundamental abstractions (Models, Resources, Agents), rollout collection, and verification patterns that power NeMo Gym.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Architecture
:link: architecture
:link-type: doc

Explore the internal design, component boundaries, and data flow patterns that make NeMo Gym modular and scalable.
:::

:::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` Key Features
:link: features
:link-type: doc

Discover available resource servers, supported environments, and capabilities across domains like mathematics, coding, and instruction-following.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem
:link: ecosystem
:link-type: doc

Learn how NeMo Gym fits within the NVIDIA NeMo Framework and integrates with NeMo Agent Toolkit and other agent frameworks.
:::

::::