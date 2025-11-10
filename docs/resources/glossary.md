# Glossary

Essential terminology for agent training, reinforcement learning workflows, and NeMo Gym architecture. This glossary organizes terms by category for easy reference.

---

## Architecture & Components

```{glossary}
Policy Model
  The primary LLM being trained or evaluated—the "decision-making brain" you want to improve.

Orchestration
  Coordination logic that manages when to call models, which tools to use, and how to sequence multi-step operations.

Verifier
  Component that scores agent outputs, producing reward signals. May also refer colloquially to "training environment with verifiable rewards."

Service Discovery
  Mechanism by which distributed NeMo Gym components find and communicate with each other across machines.

Reward
Reward Signal
  Numerical score (typically 0.0–1.0) indicating how well an agent performed on a task.

Resource Server
  Service that provides tools (functions agents can call) and verification logic (scoring agent performance).

Responses API Model
  Model implementation that follows OpenAI's Responses API format for handling agent interactions.

Responses API Agent
  Agent implementation that orchestrates between models and resource servers using the Responses API format.
```

---

## Technical Infrastructure

```{glossary}
Responses API
  OpenAI's standard interface for agent interactions, including function calls and multi-turn conversations. NeMo Gym's native format.

Chat Completions API
  OpenAI's simpler interface for basic LLM interactions. NeMo Gym includes middleware to convert formats.

vLLM
  High-performance inference server for running open-source language models locally. Alternative to commercial APIs.

NeMo Gym CLI
  Command-line interface (`ng_run`) for starting servers, managing configurations, and running rollout collection.

Global Config
  Central configuration file (`.nemo_gym_global_config.yaml`) that maps server names to URLs for service discovery.
```

---

## Interaction Patterns

```{glossary}
Multi-turn
  Conversations spanning multiple exchanges where context and state persist across turns.

Multi-step
  Complex tasks requiring agents to break problems into sequential steps, often using tools and intermediate reasoning.

Tool Use
Function Calling
  Agents invoking external capabilities (APIs, calculators, databases) to accomplish tasks beyond text generation.

Agentic Workflow
  Multi-step processes where agents make decisions, use tools, and adapt based on intermediate results.
```

---

## Rollout & Data Collection

```{glossary}
Rollout
Trajectory
  A complete sequence of agent-environment interactions, from initial prompt through tool usage to final reward score. The complete "story" of one agent attempt.

Rollout Batch
  A collection of multiple rollouts generated together, typically for the same task. Used for efficient parallel processing.

Task
  An input prompt paired with environment setup (tools + verification). What you want agents to learn to do.

Task Instance
  A single rollout attempt for a specific task. Multiple instances per task capture different approaches.

Trace
  Detailed log of a rollout including metadata for debugging or interpretability.

Data Generation Process
  The complete pipeline from input prompt to scored rollout, involving agent orchestration, model inference, tool usage, and verification.

Rollout Collection
  The process of applying your data generation pipeline to input prompts at scale.

Demonstration Data
  Training data format for SFT consisting of input prompts paired with successful agent responses. Shows models examples of correct behavior.

Preference Pairs
  Training data format for DPO consisting of the same prompt with two different responses, where one is preferred over the other.
```

---

## Training Approaches

```{glossary}
SFT
Supervised Fine-Tuning
  Training approach using examples of good agent behavior. Shows successful rollouts as training data.

DPO
Direct Preference Optimization
  Training approach using pairs of rollouts where one is preferred over another. Teaches better vs worse responses.

RL
Reinforcement Learning
  Training approach where agents learn through trial-and-error interaction with environments using reward signals.

Online Training
  Agent learns while interacting with environment in real-time (RL).

Offline Training
  Agent learns from pre-collected rollout data (SFT/DPO).
```

---

## Quick Reference Tables

### Training Data Formats

| Format | Structure | Use Case | Training Method |
|--------|-----------|----------|-----------------|
| **Demonstrations** | Prompt + successful response | Show correct behavior | SFT |
| **Preference Pairs** | Prompt + chosen + rejected | Teach better vs worse | DPO |
| **Scored Rollouts** | Prompt + response + reward | Learn from trial-error | RL |

### Verification Patterns

| Pattern | Description | Example Domains |
|---------|-------------|-----------------|
| **Correctness** | Compare output to ground truth | Math, QA, code execution |
| **Quality** | Measure adherence to requirements | Instruction following |
| **Efficiency** | Score resource usage | Tool call minimization |
| **Hybrid** | Combine several criteria | Complex real-world tasks |

### Model Integration Options

| Integration | Purpose | When to Use |
|-------------|---------|-------------|
| **OpenAI API** | Commercial hosted models | Quick prototyping, GPT-4 |
| **Azure OpenAI** | Enterprise OpenAI access | Corporate deployments |
| **vLLM** | Self-hosted open models | Privacy, cost control, custom models |

---

## Concepts

Explore deeper explanations of how NeMo Gym works:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Core Abstractions
:link: concepts-core-abstractions
:link-type: ref

Understand how Agents, Models, and Resources remain decoupled yet coordinated as independent HTTP services, including which endpoints each abstraction exposes.
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Rollout Collection Fundamentals
:link: concepts-rc-fundamentals
:link-type: ref

Learn why complete interaction transcripts matter for reinforcement learning, how they enable evaluation, and how collection orchestrators stream results to JSONL datasets.
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Verifying Agent Results
:link: concepts-verifying-results
:link-type: ref

Explore how resource servers score agent outputs with `verify()` implementations that transform correctness, quality, and efficiency checks into reward signals.
:::

::::
