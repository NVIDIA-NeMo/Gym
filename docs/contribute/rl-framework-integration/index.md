(training-framework-integration)=

# Training Framework Integration

These guides cover how to integrate NeMo Gym into a new RL training framework. Use them if you're:

- A training framework maintainer adding NeMo Gym support
- Contributing NeMo Gym integration for a training framework that doesn't have one yet

:::{tip}
Just want to train models? Use {ref}`NeMo RL <rl-training-with-nemo-rl>` instead.
:::

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
:caption: Training Framework Integration
:hidden:
:maxdepth: 1

Generation Backend <generation-backend-and-openai-compatible-http-server>
On-Policy Corrections <openai-compatible-http-server-on-policy-correction>
Integration Footprint <gym-integration-footprint-and-form-factor>
Success Criteria <gym-rl-framework-integration-success-criteria>
```
