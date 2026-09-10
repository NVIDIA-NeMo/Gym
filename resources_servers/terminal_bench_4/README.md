# Terminal-Bench 4.0 (separate verifier)

Resources server for Terminal-Bench 4.0 tasks. TB4 grades every task in a **separate
verifier container**: the agent's container is torn down, a fresh container is started from
the task's `tests/Dockerfile` image (tests baked into `/tests`), only the files the task
declares under `artifacts = [...]` in `task.toml` are re-materialized at their original paths,
and `/tests/test.sh` writes `/logs/verifier/reward.json` or `reward.txt`. This server
reproduces that contract (harbor 0.22.0 `Trial._run_separate_verifier`) on any Gym sandbox
provider, using the **prebuilt** images the TB4 release publishes, so no image is built here.

It is deliberately separate from `resources_servers/terminal_bench_2_1`, which grades inside
the agent's own sandbox by uploading `tests/` at verify time (TB2.1's "shared" mode).

## Rows

See `task_data.py`. Each row carries `task_name`, `docker_image` (agent image),
`verifier_docker_image`, `task_folder` (holds `task.toml`, and `solution/` for oracle mode)
and optional `*_digest` provenance. `benchmarks/terminal_bench_4/prepare.py` builds rows from
a local TB4 task tree and the release tag scheme
`harborframework/terminal-bench:<task>-{environment,verifier}-<release_tag>`.

## What verify() does, in order

1. Main-service `[[verifier.collect]]` hooks run in the agent sandbox (sidecar hooks are
   recorded as skipped).
2. Declared main-service artifacts plus the convention dir `/logs/artifacts` are probed
   (`dir`/`file`/`missing`), packed into ONE gzip tarball rooted at `/` (GNU tar, or a
   Python fallback), size-checked against `artifact_max_bytes`, and downloaded.
3. The agent sandbox is stopped (`stop_agent_sandbox_before_verify`).
4. The verifier sandbox starts from `verifier_docker_image` with `[verifier.environment]`
   resources, else a copy of `[environment]` (Harbor's rule).
5. `/logs/verifier` and `/logs/artifacts` are created and `chmod 777`; every directory
   artifact's target is emptied; every file artifact's parent is created and `chmod 777`
   (Harbor's `empty_dirs`/`ensure_dirs`); the tarball is extracted with `--no-same-owner`,
   so files arrive owned by the verifier user with modes preserved, as under Harbor.
6. `/tests/test.sh` must already be in the image; it is `chmod +x` and run as
   `(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1` under the task's
   `[verifier] timeout_sec` (scaled by `verifier_timeout_multiplier`, floored, capped).
7. `reward.json` is read before `reward.txt` (`{"reward": x}`, a single numeric value, or a
   bare float). `ctrf.json` and the test stdout are mirrored into `logs_dir/<task>__<session>/`.
8. The verifier sandbox is always stopped.

Failure policy: infrastructure failures (sandbox would not start, artifacts could not be
packed/uploaded/extracted, verifier image lacks `/tests/test.sh`, compose or GPU task
refused) **raise**, so the rollout is invalidated instead of scored 0. Outcomes Harbor also
records as an ungraded trial (verifier timeout, missing or unparseable reward) return
`evaluation_completed = false`, reward 0 and a `failure_reason`.

Not supported (refused with a reason): multi-service docker-compose tasks and their sidecar
artifacts (11 of the 53 whitelisted TB4 tasks), GPU tasks, `[verifier.collect]` hooks on
sidecars. No network-policy phases are enforced.

## Quickstart

### Oracle (golden solution) check — run this before trusting any model result

```bash
gym env start \
    --config resources_servers/terminal_bench_4/configs/terminal_bench_4.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +terminal_bench_4_resources_server.resources_servers.terminal_bench_4.debug=true \
    +terminal_bench_4_resources_server.resources_servers.terminal_bench_4.is_verifying_golden_patch=true
```

Then POST each row plus an empty `response` to `/verify`; the server creates the agent
sandbox, uploads `<task_folder>/solution` to `/solution`, runs `bash /solution/solve.sh` as
the image user, and grades. Reference solutions must score 1.

### Rollouts with the OpenCode sandboxed agent

```bash
TB4_TASKS_DIR=/path/to/tb4/tasks gym eval prepare --benchmark terminal_bench_4/opencode
gym eval run --config benchmarks/terminal_bench_4/opencode.yaml \
    --config benchmarks/nemotron_3.5_super/sandbox_utils.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    ++use_absolute_ip=true ++port_range_low=63000 ++port_range_high=64000
```

## Licensing information

Code: Apache 2.0. Tasks and images: Terminal-Bench 4.0 (Laude Institute), see the task
repository's licence; task content is benchmark data and must never enter training corpora.

Dependencies
- nemo_gym: Apache 2.0
