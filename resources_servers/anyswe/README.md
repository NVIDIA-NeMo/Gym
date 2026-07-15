# anyswe

Grades SWE task patches.

The agent (e.g. `sandbox_agent` with `patch_workdir: /testbed`) captures the rollout's
`git diff` into response metadata as `model_patch`. This verifier rebuilds the task's
instance container in a fresh sandbox, applies the patch, and runs the official SWE-bench
evaluation scripts (`swebench` package, `make_test_spec` + log parsers) via
`verify_task`. Reward is 1.0 when the instance is resolved.

Dataset rows need `verifier_metadata` with `instance_id`, `test_patch`, `fail_to_pass`,
`pass_to_pass`, and `instance_dict` (the raw SWE-bench instance row), plus
`responses_create_params.metadata.docker_image` naming the instance image.

See `environments/swe` for a full wiring with the sandboxed opencode harness.
