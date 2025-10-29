---
orphan: true
---

# About NVIDIA NeMo Gym

NeMo Gym standardizes scalable rollout collection for LLM and VLM training. It provides the infrastructure to systematically generate agent interaction data and a curated collection of high-quality RL environments, making it easy to produce large-scale training data for reinforcement learning workflows using the framework of your choice.

By offering unified interfaces to heterogeneous RL environments and integrations with popular RL training frameworks (VeRL, NeMo-RL, OpenRLHF), NeMo Gym lets you focus on research and model improvement rather than infrastructure and orchestration.

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

**Unified Interfaces**  
Standard abstractions for models (LLM inference), resources (tools + verification), and agents (orchestration) enable mix-and-match composition. Connect any model to any environment using consistent APIs based on OpenAI's Responses format.

**Curated Environments**  
A growing collection of resource servers across domains (mathematics, coding, knowledge, instruction-following, agent workflows) provides both the tools agents can use and the verification logic to score their performance. Each environment includes training datasets, validation sets, and examples.

**High-Throughput Architecture**  
Async orchestration with configurable parallelism supports generating thousands of rollouts concurrently. The system handles complex coordination (model inference, tool calls, verification) while maximizing throughput for training workloads.

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

NeMo Gym serves three primary user journeys:

### Model Training Researchers

**Goal**: Train models using RL with my preferred training framework  
**Use**: NeMo Gym + RL Frameworks (VeRL, NeMo-RL, OpenRLHF, etc.)

You want to train a model to improve tool calling, reasoning, or task performance using reinforcement learning. NeMo Gym provides the infrastructure to generate training rollouts at scale from curated environments or your own custom verification logic, outputting data in formats compatible with your chosen RL framework.

### Agentic Framework Users (NAT)

**Goal**: Train models using data from my agent workflow built with NeMo Agent Toolkit  
**Use**: NAT + NeMo Gym + RL Frameworks

You have built an agent using NVIDIA NeMo Agent Toolkit (NAT) and want to improve your underlying model through RL. NAT handles trajectory collection and scoring from your agent workflow, then passes the data to NeMo Gym, which integrates with RL frameworks for model weight updates through backpropagation.

### Other Framework Users

**Goal**: Train models using data from my agent workflow built with LangChain, LangGraph, CrewAI, or other frameworks  
**Use**: Your Framework + NeMo Gym + RL Frameworks

Similar to the NAT journey, your existing agent framework collects trajectories and scores them via its evaluation system. NeMo Gym provides the bridge to RL training frameworks, handling data format conversion and training orchestration.

---

## What You Can Build

NeMo Gym enables several key use cases:

**RL Training Pipelines**  
Generate training data at scale for reinforcement learning algorithms. Collect rollouts with reward signals, export in RL-framework-compatible formats, and feed directly into training loops.

**Benchmark Evaluation**  
Systematically evaluate agent performance across tasks. Use curated datasets and verification logic to measure accuracy, reasoning quality, tool-use effectiveness, and instruction-following capabilities.

**Custom Environments**  
Build your own resource servers with domain-specific tools and verification logic. NeMo Gym's base classes and templates make it straightforward to add new environments that integrate seamlessly with the rest of the system.

**Multi-Framework Experiments**  
Compare different RL algorithms and training frameworks using the same environment and data. Swap VeRL for NeMo-RL or OpenRLHF by changing configuration without rewriting environment code.

---

## Navigating This Collection

TBD.