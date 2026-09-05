---
name: add-benchmark
description: >
  Add or integrate a fixed evaluation benchmark in NeMo Gym using the current
  manifest-backed workload flow. Use when creating a benchmark catalog entry,
  reusing or adding a scorer, preparing a benchmark dataset, or porting an
  upstream benchmark. Do not use for a training-only environment or a
  component-only resources server.
---

# Add a Benchmark

## Read the current contract

Before editing, read these pages and follow their links only as needed:

- `fern/versions/latest/pages/contribute/environments/new-environment.mdx` for manifests, integration profiles,
  validation layers, and publish readiness.
- `fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx` for benchmark-specific requirements.
- The data-preparation and prompt-configuration pages linked from the benchmark guide.

Treat the current CLI, Pydantic schemas, and generated scaffold as authoritative when they disagree with prose.

## Preserve the upstream contract

Record the upstream revision, license, canonical splits, task identifiers, prompt templates, scoring behavior, reward
range, and published reference metrics before choosing the integration. Reproduce upstream task-level results where
practical; aggregate parity alone can hide conversion and scoring errors.

Choose the narrowest integration profile that preserves the workload:

- `custom-gym-verifier`: Gym owns rollout orchestration and implements or reuses the scorer.
- `custom-gym-agent-loop`: the benchmark needs a custom Gym-hosted interaction loop.
- `external-agent-loop`: an external harness owns the loop but Gym owns model and resource services.
- `external-rollout-driver`: the external system owns the complete rollout lifecycle.

Do not create a custom agent merely to duplicate a built-in harness. If the request is for a training-only workload or
an independently reusable resources server, use that workload's contract instead of presenting it as a benchmark.

## Scaffold from the manifest

Search for reusable components first, then generate the workload skeleton:

```bash
gym search resources-servers "<scoring behavior>"
gym search benchmarks "<task domain>"

gym env init --benchmark <name> --profile custom-gym-verifier \
  --reuse-verifier <scorer> --reward-range <low> <high> --higher-is-better
```

Omit `--reuse-verifier` when adding a new scorer. Select another `--profile` when the upstream benchmark owns more of
the rollout loop. Inspect `gym env init --help` rather than guessing flags.

## Implement the authored and runtime contracts

- The manifest owns catalog metadata, integration profile, reward semantics, provenance, and publish-readiness fields.
- Hydra config remains authoritative for runtime composition: datasets, agents, models, resources servers, and wiring.
- Keep stable task IDs. Put private answers and scorer inputs in `verifier_metadata`, not prompts.
- Declare `canonical_split`, `standard_prompt_config`, and `prepare_script` when the benchmark contract requires them.
- Provide a deterministic `VERIFIER_FIXTURE` for a custom verifier and test both accepted and rejected behavior.
- Do not assume rewards are binary. Implement the declared range and optimization direction.
- Preserve Responses API items, session cookies, and trace context across multi-turn or external integrations.
- Pin or bound new dependencies, document non-Python prerequisites, and keep Apache-2.0 license compatibility.

For async HTTP, subprocesses, optional fields, and external-tool installation, follow the repository-level
`AGENTS.md`; do not copy those evolving rules into the skill.

## Validate in layers

Run the generated contract checks before broader tests:

```bash
gym env validate <name> --kind benchmark
gym env test <name> --kind benchmark
```

Add focused tests for conversion, scoring boundaries, malformed model output, failure handling, and state isolation.
Coverage must remain at least 96%, and tests must assert observable behavior.

For behavior-changing environment or agent code, run representative real smoke rollouts and inspect both agent and
verifier behavior. A full evaluation, reward profile, or training run is stronger evidence but is not a universal merge
compute gate. Compare a sample of Gym tasks and rewards with upstream fixtures or outputs before claiming parity.

Before handoff, run scoped pre-commit checks, then the repository checks required by `AGENTS.md`. New source files need
the NVIDIA SPDX header, and commits need DCO sign-off.
