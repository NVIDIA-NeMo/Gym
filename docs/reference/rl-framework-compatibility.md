---
description: "NeMo Gym compatibility with RL training frameworks (NeMo RL, TRL, Unsloth)"
categories:
  - documentation
  - reference
tags:
  - neMo-rl
  - neMo-gym
  - trl
  - unsloth
  - compatibility
content_type: reference
---

(rl-framework-compatibility)=

# RL Framework Compatibility

Reference for NeMo Gym version compatibility with supported training frameworks.

:::{seealso}
{doc}`/about/ecosystem` for training framework integrations and {doc}`/training-tutorials/nemo-rl-grpo/index` for NeMo RL GRPO training with NeMo Gym.
:::

---

## NeMo RL Container

The following table maps NeMo Gym versions to compatible NeMo RL containers. Use the latest version when possible; the table provides historical compatibility for users who cannot upgrade.

| NeMo Gym Version | NeMo RL Container | Recipe |
| --- | --- | --- |
| v0.1.1 | [`nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl/tags?version=v0.4.0) | {doc}`Nemotron 3 Nano </model-recipes/nemotron-3-nano>` |
| v0.2.0 | [`nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl/tags?version=v0.5.0) | {doc}`Nemotron 3 Super </model-recipes/nemotron-3-super>` |

---

## Unsloth

The NeMo Gym integration with Unsloth is tested on {{unsloth_pinned}}. Other versions are not guaranteed to work.

:::{seealso}
{doc}`/training-tutorials/unsloth` for the Unsloth training tutorial.
:::
