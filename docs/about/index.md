---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

NeMo Gym generates training data for reinforcement learning by capturing how AI agents interact with tools and environments.

## Core Components

Three components work together to generate and evaluate agent interactions:

- **Agents**: Orchestrate multi-turn interactions between models and resources. Handle conversation flow, tool routing, and response formatting.
- **Models**: LLM inference endpoints (OpenAI-compatible or vLLM). Handle single-turn text generation and tool-calling decisions.
- **Resources**: Provide tools (functions agents call) + verifiers (logic to score performance). Examples: math environments, code sandboxes, web search.
