---
name: nemo-gym-run-and-babysit-eval
description: >
  Use when running or recovering a NeMo Gym evaluation of a self-hosted vLLM model on Slurm.
  Compose its config, supervise the resumable chain, and assess score, coverage and health evidence.
---

# Run a NeMo Gym Evaluation to Its Result

Take the requested Slurm evaluation through completion and assess the saved evidence before reporting
its score. This skill contains both config composition and run supervision; no second skill install is
required. It currently covers self-hosted vLLM services and benchmarks already present in the pinned
Gym checkout, using the benchmark's supported agent.

## Choose the procedure

- **No config yet:** read [model configuration](references/add-model-eval-config.md), sections 1–5,
  before submitting. Use the model card and the benchmark's own config; retain the requester's serving
  parameters and ask only for missing deployment or site values. A quantized variant also follows
  section 9. Continue with the Slurm procedure in this same task.
- **Ready config or existing run:** read [Slurm execution and recovery](references/slurm.md). Use a
  supplied config or policy service as given, then validate, submit or inspect as the request requires.
- **Config-only request:** follow model configuration and validation, then hand off the resolved config
  and unrun checks. A config request alone does not request a submission or a smoke run.
- **Benchmark-specific interpretation:** read the matching [benchmark notes](references/benchmark-notes.md)
  when the Slurm procedure calls for them.

## Prerequisites

The submit config's `driver.gym_install.ref` field selects the full Gym commit to install. Follow the
installation step in the bundled Slurm procedure and inspect benchmark files at that same commit.
This release needs cluster access and a pinned Gym checkout. Local runs, user-managed API endpoints,
task-folder onboarding, and adding benchmarks or harnesses are outside its scope.

## Instructions

1. Validate the config without GPUs and inspect the generated job script. Start an evaluation with
   a small smoke (`limit: 5`, `num_repeats: 1`) before the full benchmark.
2. Submit pinned and resumable. Preserve the benchmark's prompts, grading, sampling, timeouts and
   repeat requirement. A smoke or a named subset is reported as such.
3. Monitor the saved run. Inspect with `gym eval resubmit <job id> <benchmark> --recover --inspect`;
   follow `decision.action`, `gate_task` and `operator_verdict` using the Slurm decision table.
4. Recover from saved evidence, preserving settled answers. Use the documented remedy and record
   operator actions and their reasons. A STOP without an explained remedy is reported to the operator.
5. Read coverage, aggregate metrics and quality evidence. A finished job or exit code of zero does not
   establish acceptance. Report `UNVERIFIED` with its reason; do not present it as a verified result.

Do not use plain `gym eval resubmit` or `gym job resubmit`, edit saved run files, or regenerate a settled
answer. The bundled Slurm procedure defines recovery budgets, accepted operator options and exceptions.

## Example: inspect a saved run

Use the existing run's job ID, benchmark label and agent name to read the saved evidence:

```bash
gym eval resubmit "$GYM_JOB_ID" "$BENCHMARK" --recover --inspect \
  --score-agent "$AGENT" --score-key '["agent_metrics", "mean/reward"]'
```

## Result to return

Report the Gym commit, benchmark and agent, model and serving settings, run identity and artifact
locations, task and repeat counts, aggregate score key and value, coverage and health verdict. Read the
aggregate and the scored/omitted counts before stating the score, and apply the Slurm procedure's
benchmark corrections. Include incomplete or unobserved trials, MLflow delivery state when relevant,
and recovery actions with their reasons. Acceptance requires the complete evidence and gates in the
Slurm procedure; otherwise explain what remains unresolved.
