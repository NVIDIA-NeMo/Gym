---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) is an open-source framework that generates training data for reinforcement learning by capturing how AI agents interact with tools and environments.

## Core Components

Three components work together to generate and evaluate agent interactions:

- **Agents**: Orchestrate multi-turn interactions between models and resources. Handle conversation flow, tool routing, and response formatting.
- **Models**: LLM inference endpoints (OpenAI-compatible or vLLM). Handle single-turn text generation and tool-calling decisions.
- **Resources**: Provide tools (functions agents call) + verification logic (logic to score performance). Each resource server combines both:
  - **Example - Web Search**: Tools = `search()` and `browse()`; Verification logic = checks if answer matches expected result
  - **Example - Math with Code**: Tool = `execute_python()`; Verification logic = checks if final answer is mathematically correct
  - **Example - Code Generation**: Tools = none (provides problem statement); Verification logic = runs unit tests against generated code
