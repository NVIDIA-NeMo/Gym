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

Follow the [original model-config workflow](../../../skills/nemo-gym-run-and-babysit-eval/references/add-model-eval-config.md).
For a configuration-only request, stop after configuration and validation; run a smoke only when requested.
