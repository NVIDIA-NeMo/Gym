# swe_internal_v1: internal-v1 SWE resources server

Verification for the internal-v1 SWE delivery: real GitHub PRs turned into software-engineering
tasks (Python, JavaScript/TypeScript, a few C++ repositories), each with its own prebuilt image,
a hidden test patch, a golden patch, and a vendor-authored test runner + output parser pair.

This dataset is internal and **not on the Hub**. Images are pinned to a private per-task ECR tag
(`.../sweap-pro:<instance_id>`); the delivery's image manifest TSV records which builds reached
`ready`. `data/swe_internal_v1_training.jsonl` is built offline and gitignored (rows embed patch,
test and script content).

## Grading follows the vendor harness contract

Every task carries `run_script.sh` (how to run the whole suite, or a selected list of test files)
and `parsing_script.py` (turn the suite's stdout/stderr into `{"tests": [{"name", "status"}]}`),
plus the `FAIL_TO_PASS` / `PASS_TO_PASS` ids in the parser's naming convention (`<file> | <test>`).
This is exactly what the `swe_agents` harness' `NVInternalDatasetProcessor` grades with, and
`verification.py` reproduces it step for step inside a fresh sandbox of the task's image:

1. `export` every `ENV` line of the task's Dockerfiles (`env_exports`),
2. `git reset --hard <base_commit>`, apply the candidate patch with `git apply --reject`,
3. install the hidden tests: the row's `git checkout <fix_commit> -- <test files>` command first,
   the `test_patch` diff as a fallback,
4. `bash run_script.sh <comma-separated test files>` then `parsing_script.py stdout stderr out.json`,
5. resolved iff `FAIL_TO_PASS ∪ PASS_TO_PASS` is non-empty and every id is `PASSED`.

`evaluation_completed` is false when there is no verdict to read (hidden tests could not be
installed, the parser wrote no result file, missing workdir), so an infra fault is never read as
a hard task. A `git apply --reject` exit code of non-zero that still changed the tree counts as a
partial apply and the verdict stands; a patch that changed nothing cannot claim a green suite.

Everything else is the swemer_v1 flow: per-row image, `seed_session` starts the agent's sandbox
at `/app` (blob restore if the image's `.git` was stripped, git scrub via
`resources_servers/swebench/anti_cheat.py`, committer identity seeded), the model patch is
captured with `resources_servers/swebench/patch_capture.py` (`worktree` default, `committed`
optional), model edits to hidden test files are dropped, and verification happens in a second,
clean sandbox.

## The data

| Field | Source |
|---|---|
| `instance_id` | vendor instance id (`instance_<org>__<repo>-<fix commit>`; equals the image tag) |
| `image_ref`, `nydus_ref` | manifest columns |
| `base_commit`, `solution_commit` | `base_commit_hash`, `commit_hash` |
| `patch`, `test_patch` | `gold_patch`, `test_patch` |
| `run_script`, `parsing_script` | `run_script.sh`, `parsing_script.py` |
| `test_files` | `selected_test_files_to_run` |
| `test_patch_checkout_cmd` | last line of `before_repo_set_cmd` |
| `env_exports` | `ENV` lines of `base_dockerfile` + `instance_dockerfile` |
| `FAIL_TO_PASS`, `PASS_TO_PASS` | `fail_to_pass_select`, `pass_to_pass_select` |
| `problem_statement` | the vendor's issue description + requirements (+ new interfaces) |
| `language`, `repo`, `license`, `original_issue_url`, `issue_categories`, `issue_specificity`, `evaluation_time` | provenance |

`workdir` is always `/app`. The `_with_prompt_template.jsonl` variant appends the shared
"Rules for how you solve it" block the other SWE reward-profiling jsonls carry.

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.

```bash
gym env start \
  --config resources_servers/swe_internal_v1/configs/swe_internal_v1.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/swe_internal_v1/apply_golden_patch.py \
  +training_jsonl=resources_servers/swe_internal_v1/data/swe_internal_v1_training_raw.jsonl \
  +output_jsonl=results/swe_internal_v1_golden_patch_3x/passes/pass_1.jsonl +concurrency=256

python resources_servers/swe_internal_v1/aggregate_golden_patch.py \
  +runs=results/swe_internal_v1_golden_patch_3x/passes \
  +output_jsonl=results/swe_internal_v1_golden_patch_3x/verdicts.jsonl
```

A row is supported when its golden patch resolved in every pass of a 3x sweep; the training jsonl
holds only those rows.

## Running an agent

```bash
gym env start --config resources_servers/swe_internal_v1/configs/swe_internal_v1_opencode.yaml
```
