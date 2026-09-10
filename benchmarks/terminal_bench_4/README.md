# terminal_bench_4

Terminal-Bench 4.0 through the OpenCode sandboxed agent and the `terminal_bench_4` resources
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
