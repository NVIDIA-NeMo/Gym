# Skill card: nemo-gym-analyzing-results

| Field | Value |
| --- | --- |
| Name | `nemo-gym-analyzing-results` |
| Version | 0.1 (draft, not yet reviewed or published) |
| License | Apache-2.0 (same as NVIDIA-NeMo/Gym) |
| Owner / PIC | TBD: Adam |
| Repository | NVIDIA-NeMo/Gym, `skills/nemo-gym-analyzing-results` |
| Agents it is written for | Claude Code and Codex (plain SKILL.md, no agent-specific syntax) |

## What it does

Reads, compares and explains finished Gym evaluation results: whether two numbers are comparable at all, whether a difference is real (confidence intervals, paired comparison), why a score differs from a reference, and which tasks fail and why, with evidence from the rollouts.

## Who uses it

People who compare models, quantized exports, harnesses or runs, and the coding agents that write up those results.

## Inputs and outputs

- Input: one or more finished run directories, and a reference score if there is one.
- Output: a comparability verdict, a comparison with intervals, a failure breakdown with evidence, and a short write-up.

## Requirements

- A NeMo Gym install (`pip install` or a checkout); the skill does not rely on files that exist only in a checkout.
- Network access to whatever the user's own run needs (model endpoint, benchmark data). The skill adds none of its own.

## Limits

- Not for runs that crashed or produced no rollouts (use `nemo-gym-debugging`).
- Not for running the evaluation (use `nemo-gym-running-evals`).
- It does not explain the profiling files (use `nemo-gym-reward-profiling`).

## Safety

- The skill contains instructions and reference text only; it ships no scripts and no credentials.
- It reads finished run directories only; it needs no credentials and sends nothing anywhere.
- Commands it recommends that start jobs or change state are the user's own `gym` commands, run with the user's own account and cluster access.

## Evaluation

- `evals/trigger-queries.json`: queries that should and should not trigger the skill.
- `evals/evals.json`: draft task prompts with expected behavior. The with-skill versus without-skill comparison on Claude Code and Codex has not been run yet.
- `BENCHMARK.md` is produced by the NVSkills CI pipeline after `/nvskills-ci`; it does not exist yet.
