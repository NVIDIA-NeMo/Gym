# swemer_oml: OML SWE-bench Extended resources server

Verification for the OML SWE-bench Extended delivery: 5,000 real GitHub PRs turned into
software-engineering tasks (Java, JavaScript/TypeScript, Rust, C/C++), each with its own prebuilt
image, a hidden test patch, a golden patch, and, unlike the Swemer deliveries, its own grader.

This dataset is internal and **not on the Hub**. Images are pinned to a private per-task ECR tag
(`.../ext-oml-swe-bench:<task>`); the delivery's image manifest TSV records which
builds reached `ready` (4,694 of 5,000; the rest failed to build and have no pullable image).
`data/swemer_oml_training.jsonl` is built offline from the delivery folder and its image manifest
and is gitignored (rows embed patch, test and grader content).

## Grading runs the package's own verifier

Every task ships `tests/test.sh`, `tests/config.json`, `tests/grade.py` (and `tests/test.patch`).
`test.sh` restores the graded test surface from the image's immutable `/opt/pristine` snapshot,
installs the hidden test patch fail-closed, runs the suite, and calls `grade.py`, which enforces
that every `FAIL_TO_PASS` id is individually observed passing in the framework's own structured
output and writes `1` or `0` to `/logs/verifier/reward.txt`.

The delivery spans js / maven / gradle / ctest / cargo-nextest and a long tail of ~30 framework
labels, and `grade.py` is not one file (1,510 distinct copies across the 5k tasks), so swemer_v2's
per-framework output parsing does not transfer. `verification.py` therefore stages the four
`tests/` files at `/tests` in a fresh sandbox of the task's image, applies the candidate patch
with the same fallback chain the package's `solution/solve.sh` uses, runs `bash /tests/test.sh`,
and reads the reward file back. `reward` is that file's value; `evaluation_completed` is false
when the grader produced no verdict (hidden test patch failed to install, `test.sh` failed closed,
missing workdir), so an infra fault is never read as a hard task. The apply chain's exit code is
recorded as `patch_applied`, but a non-zero exit that still changed the tree (`patch` applies the
code hunks and fails on a lockfile the image's `npm install` drifted) counts as a partial apply
and the grader's verdict stands; only a patch that changed nothing is refused a resolved. Java
rows get swemer_v1's Maven Central mirror (`/root/.m2/settings.xml` + Gradle init script): the
first smoke hit HTTP 429 from Central without it.

Everything else is the swemer_v2 flow: per-row image, `seed_session` starts the agent's sandbox
(git scrub via `resources_servers/swebench/anti_cheat.py`, committer identity seeded), the model
patch is captured with `resources_servers/swebench/patch_capture.py` (`worktree` default,
`committed` optional), model edits to hidden test files are dropped, and verification happens in
a second, clean sandbox.

## The data

`data/swemer_oml_training.jsonl` is a fixed, self-contained row-per-task JSONL built offline
from the delivery folder and its image manifest, distributed offline and gitignored (rows embed
patch, test and grader content from an internal dataset). There is no prepare script here and no
reference to any internal filesystem path, same as swemer_v1/v2. The prompt is `instruction.md`,
the delivery's agent-facing statement (a `## Description` / `## Expected Behavior` spec followed
by the user-voice narrative that `environment/prompt_statement.md` holds on its own);
`swemer_oml_training_with_prompt_template.jsonl` appends the shared "Rules for how you solve it"
block the other SWE reward-profiling jsonls carry.

| Field | Source |
|---|---|
| `instance_id` | task directory name (unique within the delivery; equals the image tag) |
| `image_ref`, `nydus_ref` | manifest columns |
| `patch` | `solution/golden.patch` |
| `test_patch`, `test_sh`, `config_json`, `grade_py` | `tests/` |
| `test_framework`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `f2p_synthetic` | `tests/config.json` |
| `problem_statement` / `prompt_statement` | `instruction.md` / `environment/prompt_statement.md` |
| `language` | `[metadata].language`, else the first `[task].keywords` entry, folded to javascript/typescript, java, rust, c/c++ |
| `repo`, `difficulty`, `task_type`, `pass_at_k_*`, `*_timeout_sec` | `task.toml` |
| `base_commit` | the `BASE=` / `git checkout` sha in `environment/Dockerfile` |

`workdir` is always `/workspace/repo`: `test.sh` hardcodes it even for the 52 tasks whose
`task.toml` says `/workspace`. The jsonl holds only the rows whose golden patch resolved in all
three passes of the 3x sweep below (`data/supported_instance_ids.txt`).

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.

```bash
gym env start \
  --config resources_servers/swemer_oml/configs/swemer_oml.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/swemer_oml/apply_golden_patch.py \
  +training_jsonl=resources_servers/swemer_oml/data/swemer_oml_training.jsonl \
  +output_jsonl=results/swemer_oml_golden_patch_3x/pass_1.jsonl +concurrency=256
```

`aggregate_golden_patch.py` joins repeated passes into supported / flaky / broken / inconclusive
buckets (an infra fault with no verdict stays separate from a nondeterministic test).

The 2026-09-29 3x sweep of all 4,694 ready rows (256 concurrent, ~2h15m per pass on one cpu node,
`results/swemer_oml_golden_patch_3x/`) resolved 92.0-92.1% in every pass:

| verdict | rows | |
|---|---:|---|
| supported | 4,294 (91.5%) | resolved in all three passes; this is `data/swemer_oml_training.jsonl` |
| flaky | 55 (1.2%) | resolved in some passes only |
| broken | 336 (7.2%) | never resolved |
| inconclusive | 9 (0.2%) | fewer than three verdicts (sandbox ended mid-run / no reward file) |

By language: JS/TS 96.2%, C/C++ 97.5%, Rust 94.6%, Java 74.6%. Java carries 239 of the 336 broken
rows: 57 images whose offline Maven/Gradle cache is missing artifacts the build needs, 39 still
hitting Maven Central 429s (Gradle plugin-portal redirects bypass the mirror), 28 other dependency
resolution failures, and 114 whose suite genuinely fails under the package's own grader. Across all
languages 187 rows fail their own grader with the golden patch applied, 14 pass the suite but
score 0 (grader/test-id mismatch), and 4 golden patches do not apply at all.

`data/swemer_oml_training_raw*.jsonl` keep all 4,694 ready rows; `data/supported_instance_ids.txt`
is the filter. A per-task JSON report of every excluded row (failure category, per-pass evidence,
grader output tail) and of the 306 images that never built sits next to the pass files.

## Running an agent

```bash
gym env start \
  --config resources_servers/swemer_oml/configs/swemer_oml_opencode.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```
