# Skill card: nemo-gym-running-evals

| Field | Value |
| --- | --- |
| Name | `nemo-gym-running-evals` |
| Version | 0.1 (draft, not yet reviewed or published) |
| License | Apache-2.0 (same as NVIDIA-NeMo/Gym) |
| Owner / PIC | TBD: Adam |
| Repository | NVIDIA-NeMo/Gym, `skills/nemo-gym-running-evals` |
| Agents it is written for | Claude Code and Codex (plain SKILL.md, no agent-specific syntax) |

## What it does

Takes an agent from "run model Y on benchmark Z" to a result that can be trusted: install Gym, write the model config (vLLM service, reasoning toggle, parsers), smoke a small run, submit locally or on Slurm, resume and recover interrupted runs, and check health and acceptance before reporting a score.

## Who uses it

People who run evaluations with NeMo Gym on an API endpoint, a local vLLM server or a Slurm cluster, and the coding agents that work for them.

## Inputs and outputs

- Input: a model (endpoint or checkpoint), a benchmark name, and the target (local or Slurm).
- Output: a run config, a submitted and inspected run, and a report that names coverage, Gym version and health.

## Requirements

- A NeMo Gym install (`pip install` or a checkout); the skill does not rely on files that exist only in a checkout.
- Network access to whatever the user's own run needs (model endpoint, benchmark data). The skill adds none of its own.

## Limits

- Not for building a benchmark or harness (use `add-benchmark` or `add-harness`).
- Not for reading or comparing finished results (use `nemo-gym-analyzing-results`).
- Slurm guidance assumes the `gym eval` commands available in the installed Gym version; whether Slurm belongs in the catalog version of this skill is an open question.

## Safety

- The skill contains instructions and reference text only; it ships no scripts and no credentials.
- Endpoint URLs and keys are read from the user's own `env.yaml`; the skill does not ask for, store or send them anywhere.
- Commands it recommends that start jobs or change state are the user's own `gym` commands, run with the user's own account and cluster access.

## Evaluation

- `evals/trigger-queries.json`: queries that should and should not trigger the skill.
- `evals/evals.json`: draft task prompts with expected behavior. The with-skill versus without-skill comparison on Claude Code and Codex has not been run yet.
- `BENCHMARK.md` is produced by the NVSkills CI pipeline after `/nvskills-ci`; it does not exist yet.
