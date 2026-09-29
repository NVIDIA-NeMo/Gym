# harness_agent

Runs any gym agent harness in a sandbox with any resources server.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agnostic, not the agent config. 
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | in-box dir the agent's `repo_dir` points at, so edits land in the graded tree |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward goes in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |


Tasks with an external verifier (e.g. math) need none of these beyond an image.
