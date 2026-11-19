# SWE runner-report resources server

Verification for software-engineering tasks that are graded by **their own test runner's JSON report**: each task
records the exact test command, report format and test ids it was validated with, and the server grades the agent's
patch by running that command and reading the report. It is meant for task sets built outside the public benchmarks
(for example, tasks mined from GitHub issues whose fix and tests come from the linked pull request), where every task
ships as a prebuilt image.

The episode lifecycle is `swemer_v2`'s (this server subclasses it): the agent works in a sandbox started from the
row's `image_ref`; the anti-cheat scrub and the baseline commit run before the agent starts; the agent's patch is
captured and replayed into a fresh verifier sandbox; inconclusive verifier runs are retried on a fresh sandbox and
reported with `evaluation_completed=false` (masked), never as a silent 0.

## What differs from `swemer_v2`

| | `swemer_v2` | `swe_runner_report` |
|---|---|---|
| Test command | a shell string; an output flag is injected per framework | the exact argv recorded at validation, with `{results}` (report path) and `{specs}` (spec files) placeholders |
| Report | parsed per framework by `swe_bench_ext` | the runner's JSON report: `vitest` (Jest-compatible `testResults[].assertionResults[]`) or `mocha` (`json` reporter) |
| Test ids | dotted ids, matched with normalization | the runner's own `<spec path>::<full test name>`, exact match |
| Replayed patch | everything except the test patch's files | only edits to editable source files (`source_prefixes` × `source_suffixes`), never a test-patch path |
| Setup | none | `setup_commands` run before the baseline commit (their outputs are the agent's starting tree) and again in the verifier |
| Resolved | all required tests pass | all required tests pass **and** the test command exits 0 within `test_timeout_s` |

Requiring a clean exit matters because candidate code runs inside the test process: it can write a passing report
and then crash or hang. Like every resources server that runs the repository's own tests, grading still trusts a
report produced by a process that executes candidate code; replaying only source files, applying the hidden test
patch after the agent finishes, and grading in a fresh sandbox narrow that, but do not remove it.

## Row fields

See `task_data.py`. A row carries `instance_id`, `image_ref`, `workdir` (default `/app/repo`), `patch` (the reference
fix, for golden-patch checks), `test_patch` (the hidden tests), `test_argv`, `results_format`, `spec_files`,
`source_prefixes`, `source_suffixes`, `setup_commands`, `setup_timeout_s`, `test_timeout_s`, `FAIL_TO_PASS` and
`PASS_TO_PASS`, plus `responses_create_params` for the agent. The image must contain the repository at the task's
base commit with its dependencies installed, `git`, `bash` and `timeout`; it must not contain the hidden tests.

The test ids must be exactly what the runner reports: `<spec path relative to workdir>::<fullName>` for vitest/Jest
(`fullName`, or `title` when absent) and `<file relative to workdir>::<fullTitle>` for mocha.

## Example data

`data/example.jsonl` holds five public SWE-bench Multilingual `vuejs/core` tasks (vitest) on their public SWE-bench
evaluation images. Their FAIL_TO_PASS / PASS_TO_PASS ids were derived by running this server's verification script
in those images: a test is FAIL_TO_PASS when it fails or is missing with only the test patch applied and passes with
the gold patch, PASS_TO_PASS when it passes in both; each row then resolves with the gold patch and does not resolve
without it. `data/example_rollouts.jsonl` holds the golden-patch verifications of those rows (no model involved).

The `mocha` format needs a mocha whose `json` reporter honors `--reporter-option output=<file>` (mocha 9+); with an
older mocha, write the report with a shell redirect in `test_argv` instead.

## Golden-patch validation

```bash
gym env start \
  --config resources_servers/swe_runner_report/configs/swe_runner_report_example.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
```

`swe_runner_report_golden_patch_resources_server` grades each row's own `patch`; a row whose golden patch does not
resolve is a broken row and should be dropped before training.

## Training

`configs/swe_runner_report_opencode.yaml` pairs the server with `opencode_sandboxed_agent`; rows come from the
training recipe. Issue images, when a task has them, go in the row's user turn as `input_image` parts, which the
OpenCode agents attach to the first message.
