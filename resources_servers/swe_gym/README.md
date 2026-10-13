# SWE-Gym resources server

Verification for [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym): 2,438
SWE-bench-format Python tasks from 11 repositories (pandas, MONAI, moto, mypy, dvc, dask, modin,
pydantic, conan, hydra, bokeh), each with a prebuilt image on Docker Hub.

Every row names its own image (`docker.io/xingyaoww/sweb.eval.x86_64.<owner>_s_<repo>-<pr>`, derived
from `instance_id` in `prepare_swe_gym.py`) with the repository at `/testbed` on `base_commit` and a
conda environment named `testbed`. The per-(repo, version) eval recipe -- pytest command, the install
step the official harness re-runs, optional eval commands -- is vendored from the SWE-bench harness
fork that `swe_agents` already uses, restricted to the 200 pairs SWE-Gym contains
(`swebench_specs.py`). Vendoring keeps the grader pinned and the server clone-free at start-up.

Sandboxes come from `nemo_gym.sandbox`, so the same server runs on OpenSandbox or any other
configured provider.

## Grading

The eval script is the SWE-bench recipe: activate `testbed`, apply the candidate patch (`git apply`,
then `patch --fuzz=5` as the harness does; a patch that applies neither way is a real 0, not an infra
fault), re-run the repo's `install` step, reset the test files to `base_commit`, apply the held-out
`test_patch`, run the repo's pytest command on the touched test files, reset the test files again.

An instance resolves only when **every** `FAIL_TO_PASS` and `PASS_TO_PASS` test is observed and passing
(`PASSED` or `XFAIL`). A test missing from the parsed output counts as failed -- treating absent as
success is the standard way a broken test command scores as a resolved instance. Only the region
between the output markers reaches the parser, so install-time noise never grades.

For mypy the touched `[case name]` keys from the test patch select the data-driven suites (its
`test_cmd` ends in `-k` for that reason); every other repo takes the touched test files.

## What the agent can and cannot see

The agent works in a sandbox created by `seed_session` from the row's image alone -- no row fields are
written into it. Before the agent gets control, the shared anti-cheat scrub
(`resources_servers/swebench/anti_cheat.py`) drops every git ref but HEAD, removes remotes, tags and
reflogs, and prunes the unreachable objects. That matters here: SWE-bench images clone the full
upstream history, so without it the fix commit and every later release tag sit in `.git` within reach
of `git log --all`. The `testbed` env is put first on `PATH` so the agent's shells run the per-instance
interpreter rather than the conda base one the image defaults to.

Grading happens in a **second** sandbox created from the same image and seeded with the eval script,
the candidate patch and the test patch. The golden patch, the test patch and the FAIL_TO_PASS /
PASS_TO_PASS lists live in the server process and the training jsonl, never in the agent's sandbox.
The prompt is `problem_statement` alone; `hints_text` (maintainer comments that often spell out the
fix) is dropped at prepare time.

## Prepare the data

```bash
# every Hub row, as input to the golden-patch sweep
python resources_servers/swe_gym/prepare_swe_gym.py --raw
# only rows whose golden patch resolved in every pass (data/supported_instance_ids.txt)
python resources_servers/swe_gym/prepare_swe_gym.py
```

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.

```bash
gym env start \
  --config resources_servers/swe_gym/configs/swe_gym.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/swe_gym/apply_golden_patch.py \
  +training_jsonl=resources_servers/swe_gym/data/swe_gym_training_raw.jsonl \
  +output_jsonl=results/swe_gym_golden_patch/pass_1.jsonl +concurrency=64
```

Repeat for three passes, then `aggregate_golden_patch.py +runs=results/swe_gym_golden_patch` buckets
rows into supported / flaky / broken / inconclusive, keeping an infra fault (no verdict) apart from a
nondeterministic test. `data/supported_instance_ids.txt` is the supported bucket of one such sweep
(2026-10-08, 3 passes: 2,278 supported, 7 flaky, 115 broken, 0 inconclusive out of the 2,400 rows that
have an image; 37 of the Hub's 2,438 rows name images that are not on Docker Hub -- 15 pandas, 6 dvc, 4 dask,
4 conan, 2 pydantic, 2 modin, 2 hydra, 1 MONAI, 1 bokeh -- and one pandas row never returned a verdict to the
client, so those 38 are excluded up front rather than left to time out). The broken rows concentrate in modin
(33 of 105) and pandas (63 of 721), where the hidden tests error at setup in the published images.
`prepare_swe_gym.py` writes only the supported rows. On a Slurm cluster where OpenSandbox is only reachable
from inside, `temp/temp_launch_swe_gym_golden_patch_full.sh` runs all of this on the cpu partition.

The sweep only shows that the golden patch is sufficient. The matching negative control grades the same
rows with no patch at all, and every row must come back unresolved:

```bash
python resources_servers/swe_gym/apply_golden_patch.py \
  +training_jsonl=resources_servers/swe_gym/data/swe_gym_training.jsonl \
  +output_jsonl=results/swe_gym_empty_patch/pass_1.jsonl +concurrency=64 +empty_patch=true
```

On 2026-10-08 all 2,278 supported rows came back unresolved with an empty patch (the test patch applied and
pytest ran on the touched files in every sandbox; every FAIL_TO_PASS test failed on the base commit), so no
row in `data/supported_instance_ids.txt` scores a pass without a fix.

`data/example_rollouts.jsonl` holds the five tasks of `data/example.jsonl` graded through this same path with their
golden patch (one row per task, `reward` 1.0), which is what the repository's data validation expects next to the
example metrics.

## Running an agent

```bash
gym env start --config resources_servers/swe_gym/configs/swe_gym_opencode.yaml
```

## Tests

```bash
gym env test --resources-server swe_gym
```
