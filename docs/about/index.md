---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

NeMo Gym provides infrastructure to generate and evaluate agent behavior at scale. It captures complete records of how AI agents interact with tools and environments—what they try, what works, and how well they perform—then transforms these records into training data for reinforcement learning workflows using the framework of your choice.

By offering unified interfaces to heterogeneous RL environments and seamless data handoff to popular RL training frameworks (VeRL, NeMo-RL, OpenRLHF), NeMo Gym lets you focus on research and model improvement rather than infrastructure and orchestration.

---

## What Are Agent Interactions?

When an AI agent tackles a task, it goes through a complete interaction cycle:

1. **Receives a task**: "What's the weather in Seattle?"
2. **Decides on actions**: Determines it needs to call a weather tool
3. **Executes tools**: Calls `get_weather(city="Seattle")`
4. **Processes results**: Receives weather data from the tool
5. **Generates response**: Crafts a natural language answer for the user
6. **Gets evaluated**: A verification system scores how well it performed

This complete sequence—input, reasoning, actions, tool results, response, and score—is what we call **interaction data**. Each interaction becomes a training example that shows what the agent tried and whether it succeeded, enabling reinforcement learning algorithms to improve agent behavior over time.

<!-- TODO(@cwing-nvidia): Add diagram showing the interaction cycle and how it connects to RL training -->

---

## The Challenge: Scaling RL Training Data

Training AI agents through reinforcement learning requires massive amounts of high-quality interaction data. Teams face several obstacles:

* **Infrastructure overhead**: Building systems to coordinate models, tools, and verification logic at scale
* **Inconsistent interfaces**: Each environment requires custom integration code and data formats
* **Quality verification**: Building accurate scoring systems that distinguish good agent behavior from bad (for example, verifying a math answer is correct, checking that code actually runs, or confirming a search tool was used appropriately). Without reliable verification, training data contains incorrect reward signals.
* **Framework fragmentation**: Difficulty connecting custom environments to different RL training frameworks
* **Throughput bottlenecks**: Processing one interaction at a time is too slow for large-scale training. To generate millions of training examples, you need to process thousands of agent interactions concurrently.

These challenges slow research iteration and make it difficult to experiment with different environments and training approaches.

---

## Core Components

NeMo Gym organizes around three core abstractions. Here's how they work together in a complete interaction:

1. **Agent** receives the user request: "What's the weather in Seattle?"
2. **Model** analyzes and decides: "I should call the weather tool"
3. **Agent** routes the tool call to the weather **Resource**
4. **Resource** executes `get_weather("Seattle")` and returns the weather data
5. **Agent** sends the tool result back to the **Model**
6. **Model** generates the final response: "It's cold in Seattle—bring a jacket!"
7. **Resource** verifies the quality and assigns a reward score: 1.0

Now let's understand each component in detail:

* **Models**: LLM inference endpoints that generate text and make tool-calling decisions. Models are stateless and handle single-turn generation. Configure using OpenAI-compatible APIs or local vLLM servers.

* **Resources**: Servers that provide both tools (functions agents can call) and verifiers (logic to evaluate agent performance and assign reward signals). Examples include mathematical reasoning environments, code execution sandboxes, web search tools, and custom verification logic.

* **Agents**: Orchestration layers that connect models to resources, handle multi-turn conversations, route tool calls, and format responses consistently. Agents coordinate the interaction loop and can be extended with custom logic.

These components communicate via HTTP APIs and can run as separate services, enabling flexible deployment and scaling.

:::{seealso}
For detailed explanations of these abstractions, see {doc}`Concepts <concepts/index>`.
:::

---

<!-- ## Target Users

TODO: rewrite this section.

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

:::: -->

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

Understand the fundamental abstractions (Agents, Models, Resources), rollout collection, and verification patterns that power NeMo Gym.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Architecture
:link: architecture
:link-type: doc

Explore the internal design, component boundaries, and data flow patterns that make NeMo Gym modular and scalable.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem
:link: ecosystem
:link-type: doc

Learn how NeMo Gym fits within the NVIDIA NeMo Framework and integrates with NeMo Agent Toolkit and other agent frameworks.
:::
::::