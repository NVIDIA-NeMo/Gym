(training-nemo-rl-grpo-index)=

# RL Training with NeMo RL using GRPO

This section covers how to perform RL training, specifically using the GRPO algorithm, with the NeMo RL training library.


## Prerequisites

Before integrating Gym into your training framework, ensure you have:

- An RL training framework with policy optimization support (PPO, GRPO, or similar)
- A generation backend (vLLM, SGLang, or equivalent)
- Familiarity with OpenAI-compatible HTTP server APIs

## Integration Components

Gym integration requires implementing the following components in your training framework:

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Generation Backend
:link: generation-backend-and-openai-compatible-http-server
:link-type: doc

OpenAI-compatible HTTP server requirements and existing implementations across RL frameworks.
+++
{bdg-primary}`prerequisite`
:::

::::

## Integration Workflow

The typical integration workflow follows this sequence:

```{list-table}
:header-rows: 1
:widths: 10 30 60

* - Step
  - Component
  - Description
* - 1
  - Generation backend
  - Expose your generation engine (vLLM, SGLang) as an OpenAI-compatible HTTP server
* - 2
  - On-policy corrections
  - Implement token ID fixes to prevent re-tokenization and re-templating issues
* - 3
  - Gym integration
  - Connect Gym to your training loop using the rollout orchestration APIs
* - 4
  - Validation
  - Verify integration using the success criteria benchmarks
```

```{toctree}
:caption: NeMo RL GRPO
:hidden:
:maxdepth: 1

about-workplace-assistant.md
setup.md
gym-configuration.md
nemo-rl-configuration.md
single-node-training.md
multi-node-training.md
```
