(training-framework-integration)=

# Training Framework Integration

These guides cover how to integrate NeMo Gym into a new RL training framework. Use them if you are:

- A training framework maintainer adding NeMo Gym support
- Contributing NeMo Gym integration for a training framework that does not have one yet

:::{tip}
Just want to train models? See existing integrations:
- {ref}`NeMo RL <training-nemo-rl-grpo-index>` - Multi-step and multi-turn RL training at scale
- {doc}`TRL (Hugging Face) <../training-tutorials/trl>` - GRPO training with distributed training support
- {doc}`Unsloth <../training-tutorials/unsloth>` - Fast, memory-efficient training for single-step tasks
:::

## Existing Integrations

NeMo Gym currently integrates with the following RL training frameworks:

**[NeMo RL](https://github.com/NVIDIA-NeMo/RL)**: NVIDIA's RL training framework, purpose-built for large-scale frontier model training. Provides full support for multi-step and multi-turn environments with production-grade distributed training capabilities.

**[TRL](https://github.com/huggingface/trl)**: Hugging Face's transformer reinforcement learning library. Supports GRPO with single and multi-turn NeMo Gym environments using vLLM generation, multi-environment training, and distributed training via Accelerate and DeepSpeed. See the {doc}`TRL tutorial <../training-tutorials/trl>` for usage examples.

**[Unsloth](https://github.com/unslothai/unsloth)**: Fast, memory-efficient fine-tuning library. Supports optimized GRPO with single-step NeMo Gym environments including low precision, parameter-efficient fine-tuning, and training in notebook environments. See the {doc}`Unsloth tutorial <../training-tutorials/unsloth>` for getting started.

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

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` On-Policy Corrections
:link: openai-compatible-http-server-on-policy-correction
:link-type: doc

Fixes for on-policy training in multi-step and multi-turn scenarios to prevent train-generation mismatch.
+++
{bdg-primary}`prerequisite`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Integration Footprint
:link: gym-integration-footprint-and-form-factor
:link-type: doc

Implementation components, form factor, and reference implementations from NeMo RL.
+++
{bdg-secondary}`implementation`
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Success Criteria
:link: gym-rl-framework-integration-success-criteria
:link-type: doc

Validation criteria and benchmarks to verify correct Gym integration.
+++
{bdg-secondary}`validation`
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
  - Expose your generation engine, such as vLLM or SGLang, as an OpenAI-compatible HTTP server
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
:caption: Training Framework Integration
:hidden:
:maxdepth: 1

Generation Backend <generation-backend-and-openai-compatible-http-server>
On-Policy Corrections <openai-compatible-http-server-on-policy-correction>
Integration Footprint <gym-integration-footprint-and-form-factor>
Success Criteria <gym-rl-framework-integration-success-criteria>
```
