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
control-plane cache, and writes one Gym JSONL row per task. The row's prompt is the fixed instruction
`FIXED_INSTRUCTION` in `prepare.py` (NL2RepoBench's own task prompt, plus a line saying the task has no network), and
it does **not** inline the spec: `start.md` is uploaded into the agent sandbox's `/workspace` at `seed_session`, and
the agent reads it there. The row also carries `verifier_metadata` with the task's test commands and expected
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

1. At `seed_session` time, one untrusted agent sandbox is created with an empty `/workspace` and `start.md` written
   into it. By default (`agent_network_mode: official`) it boots from a generic agent image with open egress, so
   results there are an upper bound; `no_internet_preinstalled` boots from the task's pinned per-task image
   (dependencies preinstalled, held-out tests and the reference implementation removed) with deny-all egress except
   the model host. See [`benchmarks/nl2repobench/README.md`](../../benchmarks/nl2repobench/README.md) for every network
   mode. The sandbox's `ttl_s` must outlive the agent's whole budget plus export and grading (25200 s for the 21600 s
   agent budget in the shipped config).
2. At `verify` time, the resources server runs
   `tar czf /tmp/workspace.tar.gz -C /workspace . --exclude=.git --exclude=__pycache__ --exclude=.venv
   --exclude=node_modules` in the agent sandbox, downloads the resulting tarball, hashes it (SHA-256), and enforces
   `workspace_tar_max_bytes` as a hard cap before proceeding.
3. The agent sandbox is stopped, and a **fresh** verifier sandbox is created, from the task's pinned per-task image
   unless `verify_image_override` or `verify_image_registry_template` says otherwise (trusted network, no egress
   restriction; the image ships the test dependencies but no reference solution). Grading is attempted
   `verifier_attempts` times (default 3), each on a fresh sandbox, when the infrastructure fails, before the row is
   masked.
4. The tarball is uploaded and restored into `/workspace`. With `grading_protocol: gym_strict` (the server's default; the shipped benchmark configs set `upstream`) it is
   extracted as is; with `upstream` the agent's packaging files and its files at the held-out test paths are deleted
   first, as NL2RepoBench's own post-processor does.
5. The task's `test_commands.json` commands (e.g. `pip install -e .` followed by a `pytest` invocation) are run in
   `/workspace`. Under `gym_strict`, if a non-final command (typically an install step) fails, the remaining commands
   are skipped, `verifier_error` records which one failed, and the row is `evaluation_completed=True` with
   `tests_passed=0`. Under `upstream`, every command runs.
6. The output is parsed for pytest summaries. `gym_strict` takes the last `N passed` / `N failed` / `N error`
   occurrences; `upstream` sums every `N passed` line, as upstream does. Each defaults to 0 when absent.
7. `reward = min(tests_passed / test_case_count, 1.0)`, clamped to `>= 0.0`, where `test_case_count` is the upstream
   `test_case_count.txt` value for the task. With `grading_protocol: upstream` every row also carries `clean_reward`,
   computed from each pytest command's outer summary line only, so tests that run pytest themselves do not inflate
   it; `reward` stays the official, leaderboard-comparable score.

### Inconclusive verification is masked, not scored

A verdict that says nothing about the agent's code is returned with `mask_sample=true`, `reward=0.0` and a
`failure_kind`, so it drops out of the scored denominator instead of counting as a 0:

- no live agent session for the verify call (`session_lost`);
- the verifier sandbox could not be started after the bounded retries, or the provider reported an error while
  restoring the workspace or running a test command (`verifier_error`, or the provider's own failure kind).

A test command that times out is a scored result: the agent's code hung. A failure to collect the workspace from the
agent sandbox is not yet masked and still scores `reward=0.0` with `verifier_error` set.

### Sandbox lifecycle

A sandbox is stopped when its session ends: by `verify`, by `/close_session` or a retire (`release_session_state`),
when seeding replaces a session, or when post-start initialization fails (the directory, guard and `start.md` setup
runs before the sandbox is returned, and a failure there stops it). Sessions abandoned before any of these still rely
on the provider's `ttl_s`.

## Network posture

`enforce_agent_no_network` and `agent_network_mode` decide the agent sandbox's egress; see the benchmark README for
the table. In the no-internet modes the policy is deny-all except the resolved model-server host, and a
`provider_options.network_policy` inherited from the sandbox config may not widen it: a `defaultAction` other than
`deny` is refused at seed time.

## No oracle/golden-patch mode

NL2RepoBench ships no reference solution per task — only a hidden pytest suite and its expected passing-test count.
Unlike `deepswe`'s `validate_golden.py`, there is therefore **no golden-patch validation mode** for this resources
server: `is_verifying_golden_patch` and the multiplicative `task_cpu_multiplier` / `task_memory_multiplier` knobs from
`deepswe` have no analog here and are intentionally omitted rather than stubbed out.
