# terminal_bench_4

Terminal-Bench 4.0 through the OpenCode or mini-SWE sandboxed agent and the `terminal_bench_4` resources
server (Harbor-style separate verifier on a sandbox provider, prebuilt v4.0.0 images).

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_4` (reward from `/logs/verifier/reward.{json,txt}` written by the
  task's own `tests/test.sh` inside a fresh verifier sandbox)

Prepare rows from a local TB4 task tree (never a clone of the whole benchmark):

```bash
TB4_TASKS_DIR=/path/to/tb4/tasks gym eval prepare --benchmark terminal_bench_4/opencode
```

`prepare.py` skips multi-service compose tasks and GPU tasks because the server refuses them;
pass `prepare_script_args` (`include_compose`, `include_gpu`, `task_names`, `release_tag`,
`inventory_json`) to change that.

Both harness compositions default to 131,072 output tokens per model call, including
reasoning. Their agent allowance and policy-model sampling override share
`tb4_max_output_tokens`. For GLM-5.3 Flash on NV inference, pass
`++tb4_max_output_tokens=128000`. For other models, keep the common default.
OpenCode advertises its full context separately and reserves the output allowance
from its input budget; choose `opencode_max_context_window` for the actual backend.
The model-server sampling override enforces the ceiling on dispatched requests.

The maintained `gym-tb-training-rollout` launcher selects the NV Flash exception
automatically for `--workflow tb4`, and records the effective cap in its manifest
and configuration receipt. Explicit smaller caps remain available for qualification.
Older TB2 compositions and frozen historical run configurations are unchanged.
