---
name: add-model-eval-config
description: >
  Onboard a model for NeMo Gym evaluations and compose the complete eval config: the vLLM policy
  service, its wiring as the policy, the reasoning toggle, and each benchmark's own config from
  Gym's `benchmarks/<name>/`, then validate and smoke it. Use when a new model or checkpoint has to
  be evaluated in Gym and no eval config exists yet, including a quantized export (NVFP4, FP8) of a
  model that also runs in BF16. run-and-babysit-eval follows this skill itself when it is asked to
  run a model that has no config yet, and runs the config this skill writes. Not for adding a
  benchmark (use add-benchmark). Triggered by: "add a model", "new checkpoint", "serve this model in
  Gym", "model config", "vllm config", "enable thinking", "reasoning parser", "tool call parser",
  "compare two models fairly", "eval config", "where does the benchmark config come from", "add an
  NVFP4 variant", "serve the quantized checkpoint with the BF16 settings".
---

# Add a Model Config to Gym

Follow the canonical [model-config procedure](../../../skills/nemo-gym-run-and-babysit-eval/references/add-model-eval-config.md).
For a config-only request, compose and validate the config, read the resolved values, and report any
checks that were not run. Do not submit or monitor a run unless the user requested an evaluation or
smoke run. When they did, continue with the bundled
[Slurm run procedure](../../../skills/nemo-gym-run-and-babysit-eval/references/slurm.md) in this same task.
