---
name: nemo-gym-integrate-environment-framework
description: >-
  Adapts an environment framework such as Harbor, OpenEnv, or Prime Intellect
  Verifiers so its environments can run and compose in NeMo Gym. Use when the user
  asks to "integrate environment framework", build an environment-framework adapter,
  connect an environment hub to Gym, or integrate Harbor, OpenEnv, or Verifiers.
---

# Integrate an Environment Framework

Assume the user has selected a framework and wants its environment collection or authoring API to interoperate with Gym. Build one reusable adapter when the framework contract permits it; do not create a bespoke wrapper for every environment.

Read [Integrate external libraries](../../../fern/versions/latest/pages/environment-tutorials/integrate-external-environments.mdx) for the current seam-selection and adapter contract.

## Map both frameworks

Before choosing files, map:

- task/dataset discovery and task identity;
- reset/session lifecycle;
- action/tool and observation schemas;
- who owns the model/agent loop;
- terminal conditions;
- reward timing and semantics;
- verifier ownership;
- concurrency and isolation model;
- dependency/plugin installation;
- trajectory and token/log-prob representation.

Record unsupported or lossy mappings explicitly.

## Select the Gym seam

- **Resources-server adapter:** choose this when the external framework exposes reset/step/tools/state/reward independently of the policy loop. This preserves Gym agent interchangeability. `resources_servers/openenv/` is the reference shape.
- **Agent-server adapter:** choose this when the framework owns the harness or an inseparable end-to-end episode runner. Wrap it in async `/run` and translate its result to Gym. `responses_api_agents/harbor_agent/` and `responses_api_agents/verifiers_agent/` are reference shapes.
- **Split adapter:** use both only when there is a stable environment API worth sharing and a separate framework-specific harness. Define the HTTP/schema boundary clearly and avoid duplicated reward ownership.

Do not select the seam from the upstream name “environment.” Select it from runtime responsibility.

## Implement a reusable adapter

1. Keep framework selection and environment IDs in typed Gym config.
2. Preserve upstream task IDs and deterministic reset inputs.
3. Translate schemas at one boundary and validate both directions.
4. Preserve Gym session isolation and cookie propagation.
5. Define reward translation: terminal reward, per-step delta, cumulative reward, or external scorer result. Never sum rewards without proving their semantics.
6. Route model calls through the Gym model server when training compatibility requires token IDs/log probabilities.
7. Keep dependencies isolated in the server's `requirements.txt`; document plugin installation and version constraints.
8. Make lifecycle cleanup explicit for environments, sandboxes, workers, and artifacts.
9. Add a minimal example environment/config proving that a new upstream environment can be enabled mostly through configuration.

Use [Configuration](../../../fern/versions/latest/pages/reference/configuration.mdx) for the runnable composition and [Async patterns and performance](../../../fern/versions/latest/pages/infrastructure/async-patterns-and-performance.mdx) for implementation constraints.

## Validate compatibility

- Contract-test reset, step/tool invocation, observation conversion, termination, reward, and cleanup.
- Test at least two upstream environments with different action/reward shapes when available.
- Compare the same tasks upstream and through Gym.
- Verify task counts, trajectories, terminal state, per-task rewards, and aggregate metrics.
- Exercise concurrent sessions and confirm no state leakage.
- Run an evaluation smoke test; if online RL is a goal, also verify token/log-prob propagation with the target training framework.
- Document capability coverage and known incompatibilities.

An adapter is complete when adding another compatible upstream environment is configuration/data work, not a new integration implementation.
