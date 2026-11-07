# SWE rows to Harbor tasks

`convert.py` turns rows of the `scale_swe` and `swe_rebench` resources servers into Harbor task folders, so the
same tasks can run through the `harbor` environment server instead of `single_agent_turn` with the resources
server. The goal is the same episode on both paths: same image, same setup before the agent, same prompt, same
patch capture and the same grading code.

```bash
python scripts/swe_rows_to_harbor/convert.py rows.jsonl out/swe-tasks \
    --dataset-name nemo-gym/swe-tasks --skip-unsupported
```

Rows are JSONL, typed (`{"task_id": ..., "task_input": {...}}`) or flat. The source is the row's `source_server`,
or `--source` for rows without one. `--skip-unsupported` leaves out rows of other sources; `conversion.json` in the
output lists what was converted and skipped, with the Gym commit the code came from.

## How a row maps to a task

| Resources server (`single_agent_turn`) | Harbor task |
| --- | --- |
| Seed starts the row's image | `environment/Dockerfile`: `FROM <image>`, run without a build |
| Seed setup: `pre_commands` (scale_swe), a git repository if missing (scale_swe), the anti-cheat scrub, a committer identity, the base commit and the files already untracked | The Dockerfile's single `RUN` line, which runs `environment/setup.sh` (base64-encoded: a no-build Dockerfile cannot `COPY`). Every step stays non-fatal, as at the server |
| The row's prompt | `instruction.md`, byte for byte |
| `/verify` captures `git add -N . && git diff <base>` | A `[[verifier.collect]]` hook runs the same command in the agent's sandbox and saves the patch under `/logs/artifacts/nemo_gym` |
| Grading in a fresh sandbox of the same image | `environment_mode = "separate"` with `[verifier.environment].docker_image` set to the row's image; `tests/test.sh` runs `tests/nemo_gym/grade.py` |
| `run_verification` and its parsing | `grade.py` calls the server's own functions, copied from this checkout into `tests/nemo_gym/` under their Gym paths; its one sandbox `exec` runs locally |
| `evaluation_timeout`, sandbox resources | Read from `resources_servers/<source>/configs/<source>.yaml` (`--evaluation-timeout-sec` replaces the timeout); the verifier gets 300 s on top for the grader |
| Golden-patch check | `solution/solve.sh` applies the row's `patch`, for `gym dataset validate` |

`grade.py` writes `reward.json`: `reward` is 1.0 when the instance resolves, as the server scores it; the other keys
are metrics (`evaluation_completed`, `patch_capture_failed`, `test_patch_failed`, ...). `grading.json` keeps the
error and the per-test results, and `test_output.log` the test run.

## Requirements and limits

- The task image needs Python 3.10 or newer for the grader (SWE-rebench's parsers require it). `test.sh` looks for
  one and fails with a message when there is none.
- Gym records each image's configuration from its registry when it prepares the dataset, because it runs the
  Dockerfile's `RUN` line itself. Public images need no credentials; a private registry needs the sandbox block's
  `images.auth`.
- `swe_rebench` grading uses SWE-rebench's upstream log parsers. The converter fetches them the way the server does
  (a shallow clone of the upstream default branch, cached) and copies the files each task needs.
- Agent budget: `--agent-timeout-sec` (default 4800) sets `[agent].timeout_sec`; match it to the run being compared.
- Test-run budget: `--evaluation-timeout-sec` replaces each server's `evaluation_timeout` when the compared run
  overrides it. MR 69 sets 1200 s for every source, while `swe_rebench` defaults to 3600 s.
- The server retries a verification that reached no verdict once on a fresh sandbox; Harbor does not. Such a run
  scores 0 with `evaluation_completed = 0`.
- The base commit and the list of pre-existing untracked files live in `/var/lib/nemo_gym` in the agent's sandbox,
  where the server keeps them out of the agent's reach.
- Only the worktree patch capture mode is converted. Other SWE sources (`swemer_v1`, `swemer_v2`, `swe_next`, ...)
  follow the same pattern: an adapter in `convert.py` and a grading function in `grade.py`.
