# Swemer Agentic-v1 resources server

Verification for Swemer's Agentic-v1 SWE task packages: real GitHub PRs turned into
software-engineering tasks, each with its own prebuilt image, held-out tests, and a golden patch.

Unlike the other SWE resources servers, this dataset is **not on the Hub**: it is internal.
Task definitions live in local delivery folders (`ext-nvidia-agentic/delivery_*/tasks/<task>/`)
and built images are tracked in `swemer_v2_images.tsv` (shared with swemer_v2 -- the `repo`
column distinguishes `v1` from `v2`). There is no prepare script in this repo and no reference to
any internal filesystem path in shipped code -- `data/swemer_v1_training.jsonl` is a fixed,
self-contained row-per-task JSONL built offline, gitignored (see `data/.gitignore`: rows embed
patch and test content from an internal dataset, which is exactly what cannot be committed here).

## Grading uses swe-bench-ext, not a hand-rolled parser

swemer_v2's verifier hand-rolls a parser per framework (5 of them, extensively documented and
hardened against real captured output). v1 spans roughly 22 distinct `test_framework` values in
its raw `test_metadata.json` files -- writing and pinning a parser per framework the way v2 did
would be a large duplication of work that `responses_api_agents/swe_agents/swe_bench_ext/` has
already done: `frameworks.py` maps each framework to its structured-output flag and where the
result lands (stdout, a JSON/XML file, or a `find:`-style glob for JUnit/Maven's
`surefire-reports` layout), and `parsing.py` parses that result into `{test_id: PASSED/FAILED/
SKIPPED}`, keyed by each framework's own real node-id convention.

This is a genuine difference from v2's own data: v1's `FAIL_TO_PASS`/`PASS_TO_PASS` ids are the
framework's real node ids (pytest slash-paths, JUnit `classname::name`, ...) exactly as
`swe_bench_ext`'s parsers key their own output -- there is no id-translation step, unlike v2's
dotted-path convention (which required matching real node ids against a translated id by common
suffix; see swemer_v2's README for why that was needed there).

`verification.SUPPORTED_FRAMEWORKS` is every framework `swe_bench_ext.frameworks` has a real
config for, minus `bazel`/`jasmine` (4 rows total across a full delivery scan -- not worth a
custom eval-script path for). Unsupported-framework rows are dropped before the training jsonl is
built, same as v2.

## The data

Sourced from `swemer_v2_images.tsv` (filtered to `repo == "v1"` and `status == "ready"`) joined
against each row's `<delivery>/tasks/<task>/` folder in `ext-nvidia-agentic`:

| Field | Source |
|---|---|
| `instance_id` | `<delivery>__<task>` (task directory names collide across deliveries: 168 of 9,710 in a full scan, so the delivery prefix is load-bearing) |
| `image_ref` | the tsv's own `image_ref` column |
| `patch` | `golden.patch` |
| `test_patch` | `test.patch` |
| `problem_statement` | `prompt_statement.md` (the narrative, user-voice task description -- NOT `problem_statement.md`, which is a structured technical spec written for grading, not for an agent) |
| `test_framework`, `test_command`, `language`, `FAIL_TO_PASS`, `PASS_TO_PASS` | `test_metadata.json` |

Rows with an empty `FAIL_TO_PASS` (25 of 9,710 in a full scan) are dropped: nothing then
demonstrates the golden patch fixes anything, and `grade()` would call any patch -- including a
no-op -- "resolved". An empty `PASS_TO_PASS` is common (71% of rows) and kept as-is; it is a
regression guard, not the primary "did the patch fix something" signal.

`data/swemer_v1_training.jsonl` only holds instances whose golden patch resolved in all three
passes of a full-set 3x sweep (see Golden-patch validation below) -- the same instance ids as
`data/supported_instance_ids.txt`.

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.
A row whose golden patch does not resolve cannot be used for evaluation or training.

A full-set 3x sweep of the 9,601 raw candidates resolves 8,726 (90.9%) in every pass, holding
steady at ~91-92% across all three passes even at 2000-concurrent sandboxes. That took two rounds
of fixes to get right:

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
  --config resources_servers/swemer_v1/configs/swemer_v1.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/swemer_v1/apply_golden_patch.py \
  +training_jsonl=resources_servers/swemer_v1/data/swemer_v1_training_raw.jsonl \
  +limit=40 +concurrency=8
```

It prints a resolved rate overall and per framework, and writes one row per task.
`diagnose_failures.py` separates environment failures (OOM, disk, network, timeout, no-verdict,
zero-tests-collected) from genuine row failures. `aggregate_golden_patch.py` joins repeated
golden-patch passes (e.g. a 3x sweep) into supported / flaky / broken / inconclusive buckets,
keeping an infra fault (no verdict) separate from a genuinely nondeterministic test so neither
silently corrupts the other's label.

## Running an agent

`configs/swemer_v1_opencode.yaml` wires the opencode sandboxed agent to
`swemer_v1_resources_server`, pointed at `data/swemer_v1_training.jsonl`. It lives in a separate
file rather than folded into `swemer_v1.yaml` so golden-patch validation (which never touches an
agent) doesn't need to pull in the agent's much larger sandbox/permission config.

```bash
gym env start \
  --config resources_servers/swemer_v1/configs/swemer_v1_opencode.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```
