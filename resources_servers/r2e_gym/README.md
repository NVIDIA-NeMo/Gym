# R2E-Gym resources server

Verification for [R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset):
4,578 synthetic-issue Python tasks over real commits in ten repositories (pandas, numpy, pillow,
orange3, aiohttp, tornado, scrapy, pyramid, datalad, coveragepy), each with a prebuilt image on Docker
Hub.

Every row names its own image (`docker.io/namanjain12/<repo>_final:<commit>`) holding the repository
at `/testbed` in its pre-fix state with the project venv already first on `PATH`, the held-out tests
under `/r2e_tests` and their runner at `/testbed/run_tests.sh` (untracked). The row's
`expected_output_json` is the per-test status the fixed code produces. There is no SWE-bench style
FAIL_TO_PASS list and no test patch: the hidden tests are already in the image.

Sandboxes come from `nemo_gym.sandbox`, so the same server runs on OpenSandbox or any other
configured provider.

## Grading

The eval script is R2E-Gym's own local-evaluation recipe (`run_local_evaluation.py` plus
`DockerRuntime.setup_env` / `_calculate_reward_r2e`): apply the candidate patch with
`git apply --whitespace=fix`, excluding the image's untracked files exactly as upstream does; clear
bytecode caches; stage the runner and the tests under `/root` with `/testbed/r2e_tests` as a symlink;
run the runner; parse pytest's `short test summary info` block.

Resolution is R2E-Gym's rule: the observed statuses must **equal** the expected ones exactly -- the same
set of tests, the same status for each. A missing test, an extra test or any status change is a 0.
The expected map was produced by the real fix, so anything else means the candidate behaves
differently somewhere the tests can see. `test_results` lists what differed (`missing`, `unexpected`,
`mismatched`). A patch that does not apply is a real 0, not an infrastructure fault.

## What the agent can and cannot see

The agent works in a sandbox created by `seed_session` from the row's image alone -- no row fields are
written into it. Before the agent gets control:

1. `/r2e_tests`, `/testbed/run_tests.sh` and any R2E sidecar JSON (`expected_test_output.json`,
   `parsed_commit.json`, `syn_issue.json`, ...) are deleted. They describe the hidden tests and the
   fix. This step is not best-effort: if it fails the rollout is refused.
2. The shared anti-cheat scrub (`resources_servers/swebench/anti_cheat.py`) drops every git ref but
   HEAD, removes remotes, tags and reflogs, and prunes unreachable objects. The images keep
   `origin/master` and thousands of later commits in `.git`, and the fix commit is among them.

Grading happens in a **second** sandbox created from the same image, where the tests and the runner
are intact, seeded with the candidate patch. The golden patch and the expected statuses live in the
server process and the training jsonl, never in the agent's sandbox. The prompt is the text inside
the row's `[ISSUE] ... [/ISSUE]` tags, as R2E-Gym itself presents the task.

## Prepare the data

The Hub row has no patch; `prepare_r2e_gym.py` rebuilds the fixing commit's Python-file diff from
`parsed_commit_content` (`r2e_patch.golden_patch`, byte-identical to R2E-Gym's `ParsedCommit.get_patch`
on a 300-row sample) and drops the multi-megabyte parsed commit.

```bash
# every Hub row, as input to the golden-patch sweep
python resources_servers/r2e_gym/prepare_r2e_gym.py --raw
# only rows whose golden patch reproduced expected_output_json in every pass (data/supported_instance_ids.txt)
python resources_servers/r2e_gym/prepare_r2e_gym.py
```

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.

```bash
gym env start \
  --config resources_servers/r2e_gym/configs/r2e_gym.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/r2e_gym/apply_golden_patch.py \
  +training_jsonl=resources_servers/r2e_gym/data/r2e_gym_training_raw.jsonl \
  +output_jsonl=results/r2e_gym_golden_patch/pass_1.jsonl +concurrency=64
```

Repeat for three passes, then `aggregate_golden_patch.py +runs=results/r2e_gym_golden_patch` buckets
rows into supported / flaky / broken / inconclusive, keeping an infra fault (no verdict) apart from a
nondeterministic test. `data/supported_instance_ids.txt` is the supported bucket of one such sweep
(2026-10-08, 3 passes over all 4,578 rows: 4,553 supported, 11 flaky, 14 broken, 0 inconclusive; the
misses are single environment-sensitive tests such as aiohttp's `test_client_session_timeout_zero` and
numpy's lerp/cholesky property tests), and `prepare_r2e_gym.py` writes only those rows. On a Slurm cluster where OpenSandbox is only reachable
from inside, `temp/temp_launch_r2e_gym_golden_patch_full.sh` runs all of this on the cpu partition.

The sweep only shows that the golden patch is sufficient. The matching negative control grades the same
rows with no patch at all, and every row must come back unresolved:

```bash
python resources_servers/r2e_gym/apply_golden_patch.py \
  +training_jsonl=resources_servers/r2e_gym/data/r2e_gym_training.jsonl \
  +output_jsonl=results/r2e_gym_empty_patch/pass_1.jsonl +concurrency=64 +empty_patch=true
```

On 2026-10-08 all 4,553 supported rows came back unresolved with an empty patch (the hidden tests ran in
every sandbox and failed or errored on the pre-fix tree; one pandas row segfaults pytest before the fix),
so no row in `data/supported_instance_ids.txt` scores a pass without a fix.

`data/example_rollouts.jsonl` holds the five tasks of `data/example.jsonl` graded through this same path with their
golden patch (one row per task, `reward` 1.0), which is what the repository's data validation expects next to the
example metrics.

## Running an agent

```bash
gym env start --config resources_servers/r2e_gym/configs/r2e_gym_opencode.yaml
```

## Tests

```bash
gym env test --resources-server r2e_gym
```
