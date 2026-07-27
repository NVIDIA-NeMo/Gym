---
name: nemo-gym-create-environment
description: >-
  Creates a NeMo Gym environment from scratch: tasks, datasets, a resources server,
  tools and state, verification, agent pairing, runnable configuration, tests, and
  quality validation. Use when the user says "create environment", "new environment",
  "new benchmark", "add benchmark", "new resources server", "build environment", or
  "custom environment" and is not wrapping an existing third-party benchmark.
---

# Create a NeMo Gym Environment

Assume the user knows the capability they want to teach or measure, but not Gym's implementation pattern. Build the complete runnable environment; do not stop after writing `verify()`.

## Use the right boundary

- The resources server is the environment-side runtime. It owns shared tools, isolated per-rollout state, and verification.
- The agent harness interacts with that runtime. Prefer an existing agent under `responses_api_agents/`; create one only when the control loop is domain-specific.
- A config under `environments/<name>/config.yaml` packages datasets, resources server, agent server, and model reference into a runnable composition.
- Use `benchmarks/` only for a fixed evaluation configuration. If the request is to wrap an existing benchmark, follow the external benchmark workflow described in [Add a benchmark](../../../fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx).

Read [Environments](../../../fern/versions/latest/pages/about/concepts/environments.mdx), [Architecture](../../../fern/versions/latest/pages/about/architecture.mdx), and [Configuration](../../../fern/versions/latest/pages/reference/configuration.mdx) before choosing files or ownership.

## Start with the tasks

Ask where the data comes from before scaffolding.

1. **Existing data:** inspect its license and schema, then convert each task to Gym's Responses API JSONL format. Keep privileged scoring data outside the model-visible input.
2. **Tasks must be created:** define the target behavior, task distribution, difficulty range, failure modes, and objective completion signal first. For synthetic generation patterns, inspect `resources_servers/calendar/README.md`.

Commit at least five representative examples. Do not invent a large dataset merely to complete the code path.

Read [Prepare Data](../../../fern/versions/latest/pages/data/index.mdx) for the current row contract and dataset workflow.

## Build the environment

1. Define observable success and choose a verifier pattern before implementing tools.
2. Scaffold with `gym env init --resources-server <name>`.
3. Implement the resources server:
   - `seed_session()` creates clean per-rollout state when state is needed.
   - tool endpoints expose only environment-specific actions;
   - `verify()` scores final output or state and returns a valid `BaseVerifyResponse`;
   - malformed model output and invalid tool calls become low rewards or useful tool errors, not batch crashes.
4. Choose an existing agent harness. Use `simple_agent` for a normal Responses API tool loop. Put reusable environment tools in the resources server, not the agent.
5. Add `environments/<name>/config.yaml` with the resources server, agent reference, model reference, and datasets. Use `resources_servers/<name>/configs/` for implementation-level reusable config.
6. Document task source, licenses, verification semantics, expected dependencies, and run commands.

For implementation shapes, use [Single-step environment](../../../fern/versions/latest/pages/environment-tutorials/single-step-environment.mdx), [Multi-step environment](../../../fern/versions/latest/pages/environment-tutorials/multi-step-environment.mdx), or [Stateful environment](../../../fern/versions/latest/pages/environment-tutorials/stateful-environment.mdx). For scoring choices, read [Verification patterns](../../../fern/versions/latest/pages/build-verifiers/verification-patterns/index.mdx).

## Validate

Complete all of these:

- Unit-test passing, failing, malformed, empty, and state-isolation cases.
- Run `gym env test --resources-server <name>`.
- Validate the runnable config with `gym env validate`.
- Start the composition and collect example rollouts.
- Profile repeated rollouts on models capable of producing both successes and failures.
- Inspect actual passing and failing trajectories; a plausible aggregate score is not sufficient.
- For a training environment, demonstrate that the reward is learnable and does not reward shortcuts.

Follow [New environment](../../../fern/versions/latest/pages/contribute/environments/new-environment.mdx) for the current contribution checklist and [Evaluate](../../../fern/versions/latest/pages/evaluation/index.mdx) for rollout and profiling commands.

## Guardrails

- Use async endpoints and Gym's aiohttp-based clients.
- Do not use blocking `ray.get()` in async code.
- Pass reproducible settings through Gym config, not ad hoc environment variables.
- Keep secrets out of committed config and data.
- Do not mark an environment `verified: true` until its software, reward, and baseline evidence have been checked.
