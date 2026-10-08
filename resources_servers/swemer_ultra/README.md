# Swemer Ultra resources server

Verification for Swemer Ultra SWE task packages: the `swe_bench_ext` OTS delivery (real GitHub PRs turned
into software-engineering tasks, each with its own prebuilt image, held-out tests, and a golden patch).

Unlike the other SWE resources servers, this dataset is **not on the Hub**: it is internal. Task
definitions live in one local delivery folder
(`ext-nvidia-ots-delivery/delivery_3_13_2026/swe_bench_ext/tasks/<task>/`, 1,000 tasks) and built images
are tracked in `swebext_final_manifest.tsv` (955 `status == "ready"` rows; the other 45 failed to build).
There is no prepare script in this repo (the rows are built offline by a scratch script, as for swemer_v1); shipped code
holds no internal filesystem path -- `data/swemer_ultra_training.jsonl` is a fixed, self-contained
row-per-task JSONL built offline and gitignored (see `data/.gitignore`: rows embed patch and test content
from an internal dataset, which is exactly what cannot be committed here).

## Grading uses swe-bench-ext, not a hand-rolled parser

This server is a clone of swemer_v1, whose verifier delegates output-flag injection and result parsing to
`responses_api_agents/swe_agents/swe_bench_ext/` (`frameworks.py` maps each framework to its
structured-output flag and where the result lands -- stdout, a JSON/XML file, or a `find:`-style glob for
JUnit/Maven's `surefire-reports` layout; `parsing.py` turns that into `{test_id: PASSED/FAILED/SKIPPED}`
keyed by each framework's real node ids). swemer_v2 instead hand-rolls 5 parsers; this delivery spans 13
`test_framework` values (pytest 361, go 192, jest 159, junit 61, xctest 51, maven 41, mocha 34, vitest 24,
gtest 21, cargo-nextest 5, ctest 3, bun 2, testing 1), all in `verification.SUPPORTED_FRAMEWORKS`.

`FAIL_TO_PASS`/`PASS_TO_PASS` ids are matched exactly first and through
`swe_bench_ext.parsing.normalize_test_id` as a fallback (needed for cargo-nextest's `(N/M)` counter prefix
and for the dotted pytest ids some rows carry, e.g. `tests.pkg.test_mod.TestX::test_y`).

## The data

Sourced from `swebext_final_manifest.tsv` (filtered to `status == "ready"`) joined against each row's task
folder:

| Field | Source |
|---|---|
| `instance_id` | the task directory name (unique within this delivery; also the image tag suffix) |
| `delivery` | `delivery_3_13_2026/swe_bench_ext` |
| `image_ref` | the manifest's `image_ref` column (ECR tag `swe_bench_ext__<task>`) |
| `patch` | `golden.patch` |
| `test_patch` | `test.patch` |
| `problem_statement` | `prompt_statement.md` (the narrative, user-voice task description -- NOT `problem_statement.md`, which is the structured issue write-up) |
| `test_framework`, `test_command`, `language`, `FAIL_TO_PASS`, `PASS_TO_PASS` | `test_metadata.json` |
| `issue_statement`, `interface`, `requirements`, `rubric`, `test_metadata` | the task's other files, kept for provenance / later prompt variants |

Rows with an empty `FAIL_TO_PASS` are dropped (none in this delivery): nothing then demonstrates the golden
patch fixes anything, and `grade()` would call any patch -- including a no-op -- "resolved". An empty
`PASS_TO_PASS` is common and kept as-is; it is a regression guard, not the primary signal.

`data/swemer_ultra_training.jsonl` only holds instances whose golden patch resolved in all three passes of
the 3x sweep (see below).
`data/swemer_ultra_training_with_prompt_template.jsonl` appends the shared SWE rules block to the prompt,
as the other SWE servers' `*_with_prompt_template.jsonl` files do.

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.
A row whose golden patch does not resolve cannot be used for evaluation or training.

The 3x sweep over this delivery's 955 raw candidates (`apply_golden_patch.py` x3 + `aggregate_golden_patch.py`,
2026-10-06, 200 concurrent sandboxes, 25 min per pass) resolved 93.2% / 92.7% / 92.6% per pass; aggregated:
**883 supported (92.5%)**, 8 flaky, 62 broken, 2 inconclusive. By language: python 257/295 (87%), go 186/188,
javascript 146/155, java 79/94 (84%), cpp 89/89, typescript 66/68, swift 49/51. By framework: pytest 323/361,
go 190/192, jest 154/159, junit 45/61, xctest 49/51, maven 38/41, mocha 30/34, vitest 22/24, gtest 21/21,
cargo-nextest 5/5, ctest 3/3, bun 2/2, testing 1/1. The broken rows are image/environment faults (missing modules,
dependency drift such as numpy without `trapz`), not grading misses. The verifier itself is inherited from swemer_v1, whose full-set sweep (9,601 rows, 90.9% resolved in every pass at 2000
concurrent sandboxes) took two rounds of fixes that still apply here:

- **Maven Central rate limit.** The first sweep undercounted badly (79%, degrading pass over
  pass) because maven/junit rows -- concentrated by nothing but build-tool chance, not dataset
  quality -- hit `429 Too Many Requests` from Maven Central under concurrent load. Fixed with a
  local copy of `responses_api_agents/swe_agents/maven_mirror/` (`maven_mirror/` in this
  directory, not the shared one -- see below) that redirects Maven/Gradle to a Google-hosted
  mirror. Applied to both the verification sandbox AND the agent's own working sandbox
  (`app.seed_session`) -- an agent building/testing its own changes hits the same rate limit
  otherwise.
- **Gradle version compatibility.** The mirror script's `gradle.beforeSettings { }` registration
  is a Gradle 6.8+ API; calling it on an older Gradle throws `MissingMethodException` at
  script-evaluation time, failing the whole build outright regardless of whether dependencies
  would have resolved fine. Confirmed for real: 95 JVM rows failed with exactly this. Fixed by
  wrapping the registration in `try/catch` -- the `settingsEvaluated`/`allprojects`/
  `beforeProject` rewrite still runs as a fallback on older Gradle, just without this
  pre-resolution optimization. This fix lives only in `maven_mirror/init.gradle` here, not in
  `responses_api_agents/swe_agents/`'s copy.
- **cargo-nextest's unstable test-id counter.** Several cargo rows showed "N tests run: N passed"
  in raw output while still grading as unresolved: the dataset's stored `FAIL_TO_PASS` ids for
  cargo-nextest bake in a literal `(N/M)` progress-counter prefix (e.g. `( 4/10) mod::test`), but
  that counter reflects PARALLEL completion order, not a stable per-test identity, so it can
  differ between the run that recorded the id and any later run of the exact same test. Fixed in
  `grade()` with `swe_bench_ext.parsing.normalize_test_id` as a match fallback -- exact match
  first, normalized match only if that misses, so frameworks whose raw parser output already
  matches cleanly (pytest, go, jest/vitest, mocha) are unaffected.

```bash
gym env start \
  --config resources_servers/swemer_ultra/configs/swemer_ultra.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/swemer_ultra/apply_golden_patch.py \
  +training_jsonl=resources_servers/swemer_ultra/data/swemer_ultra_training_raw.jsonl \
  +limit=40 +concurrency=8
```

It prints a resolved rate overall and per framework, and writes one row per task.
`aggregate_golden_patch.py` joins repeated
golden-patch passes (e.g. a 3x sweep) into supported / flaky / broken / inconclusive buckets,
keeping an infra fault (no verdict) separate from a genuinely nondeterministic test so neither
silently corrupts the other's label.

## Running an agent

`configs/swemer_ultra_opencode.yaml` wires the opencode sandboxed agent to
`swemer_ultra_resources_server`, pointed at `data/swemer_ultra_training.jsonl`. It lives in a separate
file rather than folded into `swemer_ultra.yaml` so golden-patch validation (which never touches an
agent) doesn't need to pull in the agent's much larger sandbox/permission config.

```bash
gym env start \
  --config resources_servers/swemer_ultra/configs/swemer_ultra_opencode.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```
