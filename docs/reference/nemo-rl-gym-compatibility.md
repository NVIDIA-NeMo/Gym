---
description: "NeMo RL and NeMo Gym version compatibility and release timeline"
categories:
  - documentation
  - reference
tags:
  - neMo-rl
  - neMo-gym
  - compatibility
  - release
content_type: reference
---

(nemo-rl-gym-compatibility)=

# NeMo RL and NeMo Gym Compatibility

Reference for NeMo RL container versions, Nemotron model releases, and their compatible NeMo Gym versions.

:::{seealso}
{doc}`/about/ecosystem` for training framework integrations and {doc}`/training-tutorials/nemo-rl-grpo/index` for NeMo RL GRPO training with NeMo Gym.
:::

---

## Compatibility Table

| NeMo RL version | When | NeMo Gym version | Status |
|-----------------|------|------------------|--------|
| NeMo RL v0.4 container | Dec '25 | No Gym | Released |
| Nemotron 3 Nano | Dec '25 | NeMo Gym v0.1.1 | Released |
| NeMo RL v0.5 container | Jan '26 | NeMo Gym v0.1.1 | Released |
| Nemotron 3 Super | ETA Feb '26 | NeMo Gym v0.2.0 | In progress |
| NeMo RL v0.6 container | TBD: Mar '26 | NeMo Gym v0.2.0 | TBD |

---

## Version Notes

- **NeMo RL v0.4 container** — Does not include NeMo Gym; use for workflows that do not require environment infrastructure.
- **Nemotron 3 Nano** — First Nemotron model release with NeMo Gym support (v0.1.1).
- **NeMo RL v0.5 container** — Adds NeMo Gym v0.1.1 for rollout collection and training.
- **Nemotron 3 Super** — Targets NeMo Gym v0.2.0; in progress.
- **NeMo RL v0.6 container** — Planned for NeMo Gym v0.2.0; release date TBD.

---

## Choosing a Version

| Use case | Recommended |
|----------|-------------|
| NeMo Gym rollout collection and training | NeMo RL v0.5+ with NeMo Gym v0.1.1 |
| Nemotron 3 Nano training | NeMo Gym v0.1.1 |
| Nemotron 3 Super training | NeMo Gym v0.2.0 (when available) |
| RL without NeMo Gym | NeMo RL v0.4 container |
