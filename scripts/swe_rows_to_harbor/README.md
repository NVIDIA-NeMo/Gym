# SWE rows to Harbor tasks

`convert.py` converts `scale_swe` and `swe_rebench` rows into Harbor task folders with matching prompts, setup, patch capture and grading code.

```bash
python scripts/swe_rows_to_harbor/convert.py rows.jsonl out/swe-tasks \
    --dataset-name nemo-gym/swe-tasks --parser-cache-dir /path/to/frozen-parser-cache
```

Rows may be flat or typed (`task_id` plus `task_input`). Their `source_server`, or `--source`, selects the adapter.
Unsupported sources fail unless `--skip-unsupported` is explicit. `conversion.json` records all converted/skipped rows, source/Gym hashes, prompt hashes and verifier-file hashes.

| Source behavior | Harbor equivalent |
| --- | --- |
| Seed setup and task image | `environment/setup.sh` performs the source's nonfatal setup: pre-commands/repository initialization where applicable, anti-cheat scrub, Git identity and pristine-untracked tracking. A base64 Dockerfile `RUN` executes it without a build or `COPY`. |
| Original prompt | `instruction.md`, preserving exact bytes and CR/CRLF line endings. Leading canary markers and unsupported message shapes are refused. |
| Worktree patch capture | A `[[verifier.collect]]` hook runs `git add -N . && git diff <base>`, transferring the patch through `/logs/artifacts/nemo_gym`. Other capture modes are unsupported. |
| Fresh-image verification | Separate verifier mode starts the same image; `tests/test.sh` runs the copied source grading functions through `grade.py`. |
| Golden-patch control | `solution/solve.sh` applies the original patch for `gym dataset validate`. |

For SWE parity, set Harbor server options `verifier_completion_key: evaluation_completed` and `inconclusive_verification_retries: 1` (also recorded in `conversion.json`).
Incomplete verification is then masked and retried in a fresh sandbox using the same captured artifacts. Without these options, Harbor retains its default measured-zero handling of missing/invalid rewards and does not retry.
`reward.json` contains the score and completion/capture metrics; `grading.json` preserves errors and per-test results, and `test_output.log` preserves test output.

- Use the same frozen parser cache for conversion and the source server's `log_parser_cache_dir`. Conversion records its Git revision and rejects modified tracked files; a missing cache is fetched from upstream. Never independently resolve upstream main for the comparison arm.
- Pin image digests before conversion and use those same references for source rows and Harbor agent/verifier images. Dataset preparation records image configuration; private registries use the sandbox block's `images.auth`.
- Images need Python 3.10+ and Bash for grading; `test.sh` reports a missing compatible interpreter. Verifier timeouts use the image's GNU `timeout` command.
- `--agent-timeout-sec` defaults to 4800. `--evaluation-timeout-sec` overrides the source test budget; the Harbor verifier adds 300 seconds for grading overhead. Match these and the source-config sandbox resources across arms.
- Base-commit and pristine-untracked state remain under `/var/lib/nemo_gym`. Source grading modules and required upstream parser files/licenses are copied into each task; no new grader is substituted.
