---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

NeMo Gym generates training data for reinforcement learning by capturing how AI agents interact with tools and environments.

## What It Does

- **Captures agent interactions**: Records what agents try, what works, and how well they perform
- **Transforms to training data**: Converts interaction records into RL-ready datasets
- **Integrates with RL frameworks**: Seamlessly hands off data to VeRL, NeMo-RL, or OpenRLHF

**Use it for**: Training agentic models that reason with tools, interact with environments, and improve through feedback.

## Quick Start by Role

Choose your path based on your background:

**🔹 Coming from SFT?**  
Extend your synthetic data workflows to RL. {doc}`See terminology mapping → <sft-to-rl-terminology>`

**🔹 Already doing RL?**  
You'll recognize rollouts and reward functions. {doc}`See glossary → <glossary>`

**🔹 Training agentic models?**  
Jump straight to {doc}`your first agent <../get-started/first-agent>` or {doc}`tutorials <../tutorials/index>`

**🔹 New to post-training?**  
Start with {doc}`core concepts <concepts/index>`, then {doc}`setup guide <../get-started/setup-installation>`

<details>
<summary><strong>Quick Terminology Reference</strong></summary>

| Your Background | You Know | NeMo Gym Equivalent |
|-----------------|----------|---------------------|
| **SFT** | Synthetic data generation | Rollout collection |
| **SFT** | Quality filtering | Verification + rewards |
| **SFT** | Prompt templates | Agent system prompts |
| **SFT** | Training examples | Rollouts |
| **RL** | Episode | Rollout |
| **RL** | Reward function | Verifier |
| **RL** | Environment | Resource Server |
| **RL** | Policy | Policy Model |

{doc}`Full glossary → <glossary>`

</details>

## Core Components

Three components work together to generate and evaluate agent interactions:

**Models**  
LLM inference endpoints (OpenAI-compatible or vLLM). Handle single-turn text generation and tool-calling decisions.

**Resources**  
Provide tools (functions agents call) + verifiers (logic to score performance). Examples: math environments, code sandboxes, web search.

**Agents**  
Orchestrate multi-turn interactions between models and resources. Handle conversation flow, tool routing, and response formatting.

<details>
<summary><strong>Architecture Details</strong></summary>

Components communicate via HTTP APIs and run as separate services for flexible deployment:

- **Models**: Stateless, scale horizontally for throughput
- **Resources**: Serve many agents or dedicate to specific tasks
- **Agents**: Lightweight coordination layer, extend with custom logic

</details>
