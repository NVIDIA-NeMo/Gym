---
name: run-and-babysit-eval
description: >
  Take a NeMo Gym evaluation from a request to a trusted result: compose the eval config when none
  exists (by following add-model-eval-config, in the same task), submit it, let the resumable Slurm
  chain continue through walltime limits, preemption and node loss, inspect and recover the run with
  `gym eval resubmit --recover`, check health and the acceptance gates, and decide when to stop a
  chain or start a fresh run. The entry point for "run evaluations on model Y on benchmark Z", with
  or without a ready config. Not for building a benchmark or harness (use add-benchmark or
  add-harness). Triggered by: "run the eval", "run evaluations on model Y on
  benchmark Z with serving params X", "submit this benchmark", "the job timed out", "the chain
  stopped", "resume the run", "recover the run", "is the run done", "inspect the run", "gate the
  score", "restart the evaluation".
---

# Run a Gym Evaluation to Its Result

Follow the [original evaluation workflow](../../../skills/nemo-gym-run-and-babysit-eval/SKILL.md).
