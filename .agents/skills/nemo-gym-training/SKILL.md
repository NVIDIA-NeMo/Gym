---
name: nemo-gym-training
description: >-
  Configures reinforcement-learning training with NeMo Gym environments using
  supported frameworks including NeMo RL, VeRL, and Unsloth. Use when the user asks
  about "RL training", "train model", "GRPO", "NeMo RL", "VeRL", "Unsloth",
  "fine-tune with Gym", "training with environments", "RL training setup", or
  "train with my environment". SFT is outside this skill's scope.
---

# Train with NeMo Gym

Assume the user knows what model and capability they want to train. Help them connect a validated Gym environment to a supported RL framework. This workflow covers online RL, not SFT.

## Collect the required choices

Resolve these before changing configuration:

- training framework: NeMo RL, VeRL, or Unsloth;
- model/checkpoint and model-server compatibility;
- one environment or a weighted/mixed set;
- train and validation task sources;
- verifier/reward fields and expected reward range;
- compute topology and checkpoint/output location.

If the user has no framework preference, choose from the constraints documented in [RL framework compatibility](../../../fern/versions/latest/pages/reference/rl-framework-compatibility.mdx), not from generic ML preference.

## Preflight the environment

Do not start expensive training until:

1. `gym env validate` passes for every runnable config.
2. A smoke rollout succeeds through seed, agent loop, and verification.
3. Reward profiling shows both successful and unsuccessful trajectories on a capable model.
4. Actual trajectories confirm the reward measures the intended behavior and has no obvious shortcut.
5. Multi-step rollouts preserve `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs`.

Read [Training](../../../fern/versions/latest/pages/about/concepts/training.mdx) and [Training tutorials](../../../fern/versions/latest/pages/training-tutorials/index.mdx) for the supported workflow.

## Configure the selected framework

Follow the framework-specific source of truth:

- [NeMo RL GRPO](../../../fern/versions/latest/pages/training-tutorials/nemo-rl-grpo/index.mdx)
- [VeRL](../../../fern/versions/latest/pages/training-tutorials/verl.mdx)
- [Unsloth](../../../fern/versions/latest/pages/training-tutorials/unsloth.mdx)

Use the exact current configuration fields and launch commands from those pages. Wire the Gym agent/task routing explicitly; do not silently substitute a benchmark split for training data.

For multiple environments:

- start all required resources and agent servers;
- ensure every row carries the correct `agent_ref`;
- choose sampling proportions deliberately;
- scale prompts per step so no environment is starved;
- inspect metrics separately by environment as well as globally.

Read [Multi-environment training](../../../fern/versions/latest/pages/training-tutorials/multi-environment-training.mdx).

## Validate the run

- Run the smallest end-to-end training configuration first.
- Confirm rollout count, reward fields, token/log-prob fields, and optimizer updates.
- Track reward and validation metrics per environment.
- Compare against the untrained checkpoint using the same evaluation config.
- Check for reward saturation, collapse, missing environment batches, and regressions on held-out capabilities.
- Save the resolved configs, checkpoint identity, seeds, and commands needed to reproduce the run.

Do not claim learning from training reward alone. Require a fixed held-out evaluation and inspect representative trajectories.

## Guardrails

- Keep model/provider secrets in supported local secret configuration, never committed files.
- Use Gym's server and async client patterns; do not replace them with an unrelated model client.
- Separate infrastructure failures from low reward.
- Do not tune the verifier to make the training curve look better.
- Do not expand into SFT unless the user changes the request.
