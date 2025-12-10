---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

## Motivation

The agentic AI era has increased both the demand for RL training and the complexity of training environments:

- More complex target model capabilities
- More complex training patterns (e.g., multi-turn tool calling)
- More complex orchestration between models and tools
- More complex integrations with external systems
- More complex integrations between environments and training frameworks
- Scaling to high-throughput, concurrent rollout collection

Embedding custom training environments directly within training frameworks is complex and often conflicts with the training loop design.

## NeMo Gym

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) decouples environment development from training, letting you build and iterate on environments independently. It provides the infrastructure to develop agentic training environments and scale rollout collection, enabling seamless integration with your preferred training framework.

- Scaffolding and patterns to accelerate environment development: multi-step, multi-turn, and user modeling scenarios
- Contribute environments without expert knowledge of the entire RL training loop
- Test environments and throughput end-to-end, independent of the RL training loop
- Interoperable with existing environments, systems, and RL training frameworks
- Growing collection of training environments and datasets for Reinforcement Learning from Verifiable Reward (RLVR)

:::{tip}
The name "NeMo Gym" comes from historical reinforcement learning literature, where the word "Gym" refers to a collection of RL training environments!
:::

## Core Components

A training environment consists of three server components:

- **Agents**: Orchestrate the rollout lifecycle—calling models, executing tool calls via resources, and coordinating verification.
- **Models**: Stateless text generation using LLM inference endpoints (OpenAI-compatible or vLLM).
- **Resources**: Define tasks, tool implementations, and verification logic. Provide what agents need to run and score rollouts.
  - **Example - Web Search**: Task = answer knowledge questions; Tools = `search()` and `browse()`; Verification = checks if answer matches expected result
  - **Example - Math with Code**: Task = solve math problems; Tool = `execute_python()`; Verification = checks if final answer is mathematically correct
  - **Example - Code Generation**: Task = implement solution to coding problem; Tools = none; Verification = runs unit tests against generated code
