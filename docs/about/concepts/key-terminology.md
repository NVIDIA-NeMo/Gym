(key-terminology)=

# Key Terminology

Essential vocabulary for agent training, RL workflows, and NeMo Gym. This glossary defines terms you'll encounter throughout the tutorials and documentation.

## Rollout & Data Collection Terms
```{glossary}
Rollout / Trajectory
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


## Architecture Terms

```{glossary}
Policy Model
    The primary LLM being trained or evaluated - the "decision-making brain" you want to improve.

Orchestration
    Coordination logic that manages when to call models, which tools to use, and how to sequence multi-step operations.

Verifier
    Component that scores agent outputs, producing reward signals. May also refer colloquially to "training environment with verifiable rewards."

Service Discovery
    Mechanism by which distributed NeMo Gym components find and communicate with each other across machines.

Reward / Reward Signal
    Numerical score (typically 0.0-1.0) indicating how well an agent performed on a task.
```

## Training Approaches

```{glossary}
SFT (Supervised Fine-Tuning)
    Training approach using examples of good agent behavior. Shows successful rollouts as training data.

DPO (Direct Preference Optimization)
    Training approach using pairs of rollouts where one is preferred over another. Teaches better vs worse responses.

RL (Reinforcement Learning)
    Training approach where agents learn through trial-and-error interaction with environments using reward signals.

Online vs Offline Training
    - **Online**: Agent learns while interacting with environment in real-time (RL)
    - **Offline**: Agent learns from pre-collected rollout data (SFT/DPO)
```

## Interaction Patterns

```{glossary}
Multi-turn
    Conversations spanning multiple exchanges where context and state persist across turns.

Multi-step  
    Complex tasks requiring agents to break problems into sequential steps, often using tools and intermediate reasoning.

Tool Use / Function Calling
    Agents invoking external capabilities (APIs, calculators, databases) to accomplish tasks beyond text generation.
```

## Technical Infrastructure

```{glossary}
Responses API
    OpenAI's standard interface for agent interactions, including function calls and multi-turn conversations. NeMo Gym's native format.

Chat Completions API
    OpenAI's simpler interface for basic LLM interactions. NeMo Gym includes middleware to convert formats.

vLLM
    High-performance inference server for running open-source language models locally. Alternative to commercial APIs.
```
