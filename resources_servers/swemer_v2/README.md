# Swemer Agentic-v2 resources server

Verification for Swemer's Agentic-v2 SWE task packages: real GitHub PRs turned into
software-engineering tasks, each with its own prebuilt image, held-out tests, and a golden
patch.

Unlike the other SWE resources servers, this dataset is **not on the Hub**: it is internal, and
images are pinned to a private per-task ECR tag rather than a public registry. There is no
prepare script here and no reference to any internal filesystem path -- `data/swemer_v2_training.jsonl`
is a fixed, self-contained row-per-task JSONL distributed offline. Each row carries `image_ref`,
`patch`/`test_patch`, `test_framework`/`test_command`, FAIL_TO_PASS/PASS_TO_PASS,
`responses_create_params`, and `agent_ref`, and only holds instances whose golden patch resolved
in all three passes of a full-set 3x sweep (see Golden-patch validation below).

Sandboxes come from `nemo_gym.sandbox`, so the same server runs on OpenSandbox or any other
configured provider. Images pull directly from ECR (`SandboxImageSpec`/auth is not needed here —
the cluster's OpenSandbox pods have pull access to the private registry).

## Grading is framework-specific, not just language-specific

Every row ships its own hand-authored `test_command` — there is no per-language template like
Scale-SWE/SWE-rebench have. The dataset authors ran the actual suite once and recorded
FAIL_TO_PASS/PASS_TO_PASS ids in whatever format that specific invocation happened to print, so
the id format is framework-specific and sometimes inconsistent even within one framework (see
`verification.py`'s module docstring for the exact shapes observed for pytest/go/jest/mocha/
vitest).

`test_command` frequently does not already request structured output — only about half of
jest/go rows do, and mocha/vitest are mostly plain text. `inject_output_flags` forces it on
(`-rA`, `-json`, `--json`, `--reporter json`, `--reporter=json`) rather than trusting each task's
author to have set it, appending rather than inserting: every sampled command runs the test
tool as the last step in its `&&` chain, and all four CLIs here take the last occurrence of a
repeated flag.

Each of the five parsers was written and pinned against real captured output — one task per
framework, golden patch applied, run for real inside a live OpenSandbox pod against the task's
actual image — not against assumptions about a tool's JSON schema. The fixtures live in
`tests/fixtures/` (`{framework}_output.txt` + `manifest.json` for the FAIL_TO_PASS/PASS_TO_PASS
ids) and `TestFrameworkParsersAgainstRealCapturedOutput` in `tests/test_app.py` asserts each one
grades as `resolved` (a golden patch is expected to pass), which is an end-to-end check that the
flag injection and the parser agree with what actually happened. Two non-obvious things the real
captures caught that synthetic fixtures would not have:

- jest/vitest print their normal human-readable console report BEFORE the `--json`/
  `--reporter=json` blob, not instead of it, and that report can contain a bare `{` from source
  code inside a test title — so `_extract_json_object` cannot just take the first (or first
  parseable) `{`; it requires the parsed object to carry one of the report's own top-level keys
  (`testResults` for jest/vitest, `passes`/`failures`/`pending` for mocha).
- `npx`/`npm` print an update-notice banner AFTER the real JSON on the same stream, so a plain
  `json.loads` over the whole captured region fails on genuinely valid output.

A 25-row golden-patch pilot through the live resources server (not just isolated parser calls —
real sandboxes, real `verify()` responses) caught three more bugs the five single-task fixtures
above did not, because none of those five happened to exercise the shape that broke:

- **pytest class-qualified ids.** The dataset's id (`tests.pkg.mod.TestFoo::test_x`) can't be
  turned into a real node id by a fixed dots-to-slashes rule — there's no way to tell a
  module-path dot from a class-name dot in the dotted string alone. Fixed by matching the other
  direction: real node ids out of the output are flattened into the dataset's own convention
  and compared, never the reverse. 8 of 12 pytest pilot rows were false negatives from this.
- **Old pytest, no `-rA` support.** `-rA`'s `A` (all) category was only added in pytest 3.6
  (2018); a task pinned to pytest 3.3.2 silently swallowed the flag and never printed the
  summary section this parser relied on — 198/198 real passes read as 0 observed. Fixed by also
  matching plain `-v`'s inline `<id> STATUS   [ NN%]` lines whenever the summary shape doesn't
  match, not as an explicit fallback mode.
- **mocha's third id shape.** A real file-path prefix (`/workspace/repo/.../Swap.js::title`,
  distinct from both the empty-prefix `::title` and bare `title` shapes already handled) fell
  through unmatched.

Pilot resolve rate after these fixes: 22/25 (88%) — pytest 10/12, go 6/6, jest 4/4, mocha 2/3.
2 of the 3 that didn't resolve turned out to still be a parser bug, not a dataset issue as first
assessed: a `FAIL_TO_PASS` id prefix relative to a WIDER root than the real node id pytest prints
(e.g. `cd tests && pytest test_x.py`, real id `test_x.py::...`, dataset id assumes
`tests.test_x::...`) — see "More bugs, found only by full-set runs" below for the fix.
1 mocha-labeled row actually runs via Karma (a browser test runner wrapping mocha, not the mocha
CLI `inject_output_flags` targets), correctly surfaces as `evaluation_completed: false` with a
clear parse-failure `error` rather than being silently mis-graded either way — this one is a
genuine dataset/framework-label mismatch, not fixable by a parser change.

## More bugs, found only by full-set runs

Three full 2,738-row sweeps (not the 25-row pilot, and not the five single-task fixtures) found
five more real bugs — each confirmed against live full-set evidence before shipping, and each
validated offline against every affected row (not just a handful) to rule out regressions before
the next rerun:

- **pytest suffix-matching was one-directional.** The pilot's fix (matching a real node id's
  suffix against the dataset's dotted id) only handled the dataset id being SHORTER than the real
  path (a missing leading package directory). It missed the opposite: `test_command` doing
  `cd .../master && pytest ...` means the real node id never contains `master`, but the
  dataset's own id does (`master.buildbot....TestConnection::test_x`) — a dataset id LONGER than
  any real path could confirm. Fixed by matching on the longest common SUFFIX regardless of which
  side is longer, rather than requiring one side to fit inside the other.
- **A pytest teardown `ERROR` silently downgraded a real pass.** All 5 `getsentry-sentry-*` rows
  in one delivery printed every target id as both `PASSED <id>` (the test's own body ran
  correctly) and, later in the same `-rA` summary, `ERROR <id>` from an unrelated teardown-phase
  fixture (a file-descriptor-count assertion, consistent with a sandbox artifact, not anything
  the golden patch touched). Plain last-line-wins graded all of these fully-passing golden
  patches as fully unresolved. Fixed narrowly: `ERROR` never downgrades an already-recorded
  `PASSED` for the same id, but a genuine rerun `FAILED` still does (a real verdict on the test,
  unlike `ERROR`).
- **A trailing `)` corrupts a command the same way a trailing bare `exit` does.**
  `getsentry-sentry-javascript-11564-agentic-v2`'s command chains several `(cd pkg && ...)`
  subshells with `;`; appending `--json` after the last `)` is a bash syntax error regardless of
  what preceded it. Folded into the same trailing-shape guard as the bare-`exit` case.
- **`go test` appearing more than once in one command only got `-json` on the first occurrence.**
  `erigontech-erigon-14994-agentic-v2` runs `go test -v` twice (`;`-separated, one per package);
  `str.replace(..., count=1)` left the second invocation — where this row's own FAIL_TO_PASS ids
  actually lived — as plain text `_parse_go` could never match. Fixed by replacing every
  occurrence, validated safe across all 15 multi-invocation go rows in the dataset (the other 14
  already resolved via their first invocation alone, so this is purely additive for them).
- **`--outputFile=<path>` writes the JSON report to a file, not stdout.** Several vitest/jest rows
  already request `--outputFile=`; the harness only captures stdout, so the report was never
  visible to the parser regardless of which reporter flag was requested. 5 of 8 vitest rows using
  `--outputFile=` in one full-set run had no JSON anywhere in captured output for exactly this
  reason. Fixed by appending `; cat <path> 2>/dev/null || true`, safe to add unconditionally
  (including for rows whose own script already reads the file back some other way, since a
  redundant `cat` of the same content changes nothing for the parser).

The mocha fix is larger than a one-line parser change: roughly a third of mocha rows'
`test_command` wraps mocha inside the project's own runner script (`ts-node`, `jake`, a custom
`npm` script) rather than invoking the mocha CLI directly, so the appended `--reporter json`
lands on the wrapper's argv, not mocha's, and the run falls back to mocha's default `spec` text
reporter. 29 of 49 non-karma mocha rows with no JSON output in one full-set run were exactly
this: a real, fully-passing run with unambiguous per-test checkmarks. `_parse_mocha` now falls
back to scanning `✔`/`✓` checkmark lines when no JSON reporter object exists at all. Only passing
tests are extracted this way — a target id absent from the text already grades as failing, which
is correct for a title mocha never printed, so there was no need to (and no real captured
failing-block example to) parse mocha's numbered failure blocks.

## Phase 1: 5 of ~20 frameworks

The tree spans roughly 20 `test_framework` values. Only `pytest`, `go`, `jest`, `mocha`, and
`vitest` are supported (`verification.SUPPORTED_FRAMEWORKS`) — about 73% of the ~3,800 locally
available tasks. These were chosen because each has either a built-in structured-output flag or
a parser with a well-understood id format. The rest are excluded because grading them correctly
is a real open problem, not a parser someone hasn't gotten around to writing yet:

- **maven/junit** (~558 tasks): `FAIL_TO_PASS` ids are often opaque surrogate tokens like
  `"fe-core::test_7"` that do not correspond to any Maven/Gradle/JUnit-native identifier.
- **cargo** (~253 tasks): default `nextest` text output, parseable but not yet implemented.
- **gtest/ctest/cppunit and a long tail** (~220 tasks): many `test_command`s are ad hoc one-off
  scripts, not a real test-framework invocation — one example just greps the source file for a
  function name and prints `OK`/`FAIL`.

Unsupported-framework tasks are dropped before `data/swemer_v2_training.jsonl` is built, so this
server never has to grade them.

## The data

`data/swemer_v2_training.jsonl` is a fixed, self-contained row-per-task JSONL, distributed
offline (not generated by any script in this repo, and gitignored -- see `data/.gitignore`: rows
embed patch and test content from an internal dataset, which is exactly what cannot be committed
here). This is training data, not a fixed eval benchmark -- there is no frozen reference split,
and it only holds the instances whose golden patch resolved in all three passes of a full-set 3x
sweep (see Golden-patch validation below) -- the same 2,538 instance ids as
`data/supported_instance_ids.txt`.

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.
A row whose golden patch does not resolve cannot be used for evaluation or training.

```bash
gym env start \
  --config resources_servers/swemer_v2/configs/swemer_v2.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/swemer_v2/apply_golden_patch.py \
  +training_jsonl=resources_servers/swemer_v2/data/swemer_v2_training.jsonl \
  +limit=40 +concurrency=8
```

It prints a resolved rate overall and per framework, and writes one row per task.
`diagnose_failures.py` separates environment failures (OOM, disk, network, timeout, no-verdict,
zero-tests-collected) from genuine row failures. `aggregate_golden_patch.py` joins repeated
golden-patch passes (e.g. a 3x sweep) into supported / flaky / broken / inconclusive buckets,
keeping an infra fault (no verdict) separate from a genuinely nondeterministic test so neither
silently corrupts the other's label.

`data/supported_instance_ids.txt` is the result of one such sweep over the full Phase 1 set:
2,538 of 2,738 instances whose golden patch resolved in all three passes of a concurrency=1024
3x sweep. `data/swemer_v2_training.jsonl` holds exactly those rows. The other 200 are excluded
because none can be scored either way: 137 never resolve, 35 never produced a verdict in any pass
(image pull / setup failures, concentrated in mocha and jest/typescript rows), and 28 resolve
inconsistently across passes (flaky tests, concentrated in go).

## Running an agent

`configs/swemer_v2_opencode.yaml` wires the opencode sandboxed agent to
`swemer_v2_resources_server`, pointed at `data/swemer_v2_training.jsonl`. It lives in a separate
file rather than folded into `swemer_v2.yaml` so golden-patch validation (which never touches an
agent) doesn't need to pull in the agent's much larger sandbox/permission config.

```bash
gym env start \
  --config resources_servers/swemer_v2/configs/swemer_v2_opencode.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```
