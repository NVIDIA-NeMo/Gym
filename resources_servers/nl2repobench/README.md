# NL2RepoBench

This resources server implements the [NL2RepoBench](https://github.com/multimodal-art-projection/NL2RepoBench)
greenfield, "0-to-1" repository-generation benchmark. Unlike patch/diff benchmarks (e.g. `deepswe`), the coding
agent starts from an **empty workspace** plus a single natural-language spec file (`start.md`) and must write a
complete, installable Python package from scratch. There is no base repository, no git history requirement, and no
oracle/reference solution — grading is purely by running the task's hidden pytest suite against whatever the agent
produced.

## Prepare pinned task assets

```bash
python -m resources_servers.nl2repobench.prepare \
  --source-dir /path/to/NL2RepoBench \
  --no-download
```

The preparation step copies each task's four flat upstream files (`start.md`, `test_case_count.txt`,
`test_commands.json`, `test_files.json`) from `test_files/<proName>/` into the resources server's gitignored
control-plane cache, and writes one Gym JSONL row per task with the fixed instruction
("According to the start.md in the workspace, implement the entire project as per the requirements described in
start.md.") followed by the spec text, plus `verifier_metadata` carrying the task's test commands and expected
test-case count. The task's immutable per-task base image
(`ghcr.io/multimodal-art-projection/nl2repobench/<proName>:1.0`) is recorded alongside the row, and the resources
server rejects a row whose image differs from the pinned image for its task ID.

## Run OpenCode rollouts

Use the benchmark config with a model and sandbox-provider config. Launch Gym with `+use_absolute_ip=true` when
OpenSandbox needs a host-routable model-server address.

```bash
gym env start \
  --config benchmarks/nl2repobench/opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml
```

## Workspace collection and verification

Because this benchmark has no diff/commit contract to grade against, verification uses a tar-based workspace
snapshot instead of `git diff`:

1. At `seed_session` time, one untrusted agent sandbox is created from the task's pinned base image with an empty
   `/workspace`. The agent writes its package there directly (no git identity or commit step is required or seeded).
2. At `verify` time, the resources server runs
   `tar czf /tmp/workspace.tar.gz -C /workspace . --exclude=.git --exclude=__pycache__ --exclude=.venv
   --exclude=node_modules` in the agent sandbox, downloads the resulting tarball, hashes it (SHA-256), and enforces
   `workspace_tar_max_bytes` as a hard cap before proceeding.
3. The agent sandbox is stopped, and a **fresh** verifier sandbox is created from the same pinned base image (trusted
   network, no egress restriction — the base image ships the task's test dependencies pre-installed but no reference
   solution).
4. The tarball is uploaded into the fresh sandbox and extracted back into `/workspace`.
5. The task's `test_commands.json` commands (e.g. `pip install -e .` followed by a `pytest` invocation) are run
   sequentially in `/workspace`. If a non-final command (typically an install step) fails, remaining commands are
   skipped and `verifier_error` records which command failed; the run is still `evaluation_completed=True` with
   `tests_passed=0`.
6. The combined stdout+stderr of all commands is regex-parsed for the last `N passed` / `N failed` / `N error`
   pytest-summary occurrences (each defaults to 0 when absent from the summary line — that is normal, not a parse
   failure).
7. `reward = min(tests_passed / test_case_count, 1.0)`, clamped to `>= 0.0`. `test_case_count` is the upstream
   `test_case_count.txt` value for the task (the total number of tests the hidden suite is expected to run), not a
   count the agent can see or influence directly.

The untrusted agent sandbox defaults to deny-all egress (`enforce_agent_no_network`); the OpenCode benchmark config
adds only the resolved Gym model-server host to that policy.

## No oracle/golden-patch mode

NL2RepoBench ships no reference solution per task — only a hidden pytest suite and its expected passing-test count.
Unlike `deepswe`'s `validate_golden.py`, there is therefore **no golden-patch validation mode** for this resources
server: `is_verifying_golden_patch` and the multiplicative `task_cpu_multiplier` / `task_memory_multiplier` knobs from
`deepswe` have no analog here and are intentionally omitted rather than stubbed out.
