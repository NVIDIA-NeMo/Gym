---
name: nemo-gym-blade-analysis
description: >-
  Use when building, validating or reviewing a BLADE benchmark analysis
  package in NeMo Gym: the analysis skill, rollout data, golden reports with
  metrics sidecars and anchor facts, shallow baselines, and judge calibration
  with blade_toolkit.py, or when checking whether a benchmark is BLADE-ready.
  To analyze an evaluation run itself (whether a score change is real, which
  tasks failed and why), use nemo-gym-analyzing-results.
---

# NeMo Gym BLADE Benchmark Packages

## Invocation Check

Use this skill when the user wants to build, validate, submit, or review a
BLADE-ready benchmark package, or asks whether a benchmark has all required
BLADE deliverables.

The analysis method itself (outcome groups, sometimes-pass comparisons,
root-cause labels, evidence rules, and report structure) is in the
`nemo-gym-analyzing-results` skill, under `references/failure-diagnosis.md`.
Use that skill to analyze a run, compare runs, or write the diagnostic analysis
that becomes a golden report, then return here to package and check it.

Load `references/blade-benchmark-build-guide.md` for the deliverables, their
checks, and judge calibration.

Use the bundled public helper at `scripts/blade_toolkit.py` for package
validation, draft anchor-fact extraction, shallow baseline generation, and local
calibration when external BLADE tooling is not available in the target
repository.

Do not load benchmark-specific examples by default. Load
`references/cvdp-report-example.md` only when the user explicitly asks for a
CVDP example, the original CVDP report layout, or this optional reference, or
when the agent is confused about the goal and needs one concrete example to
re-anchor on what a BLADE-style report is supposed to look like.

Nemotron-only golden analysis artifacts are available under
`references/nemotron-analysis-artifacts/` as original-CVDP example artifacts.
Load them only when the user explicitly asks to study an example completed
BLADE-style report, asks for CVDP artifacts, or the agent is confused about the
goal and needs a concrete completed example. Do not load those files by default.

## Inputs To Gather

Start by identifying the artifact set:

- rollout JSONL from `gym eval run`, with its `<stem>_materialized_inputs.jsonl`
- aggregate metrics JSON (`<stem>_aggregate_metrics.json`), if present
- reward profile JSONL from `gym eval profile`, if present
- benchmark-specific report directory, if present
- optional golden analysis artifacts, if the user asks to compare against a
  curated report
- config paths, agent name, model name, repeat count, and sampling settings
- source dataset metadata, license, and known redaction limits

If artifacts are missing, state which claims cannot be supported rather than
filling gaps from memory.

First build an inventory of BLADE deliverables: analysis skill, rollout data, and golden
report packages with metrics and anchor facts. Current BLADE scoring is handled
by the universal `blade-judge`; benchmark-local judge utilities are optional
pre-checks, not required deliverables. Missing deliverables are blocking work
items, not footnotes.

If external BLADE tools are not available, use the local helper script. Paths are from the repository root:

```bash
uv run python .agents/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py validate --benchmark-dir <benchmark_dir>
uv run python .agents/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py extract-anchor-facts --help
uv run python .agents/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py make-shallow --help
uv run python .agents/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py calibrate --help
```
