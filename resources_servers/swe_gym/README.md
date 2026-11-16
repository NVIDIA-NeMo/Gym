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

`swe_gym_opencode.yaml` runs `opencode_agent` through `single_agent_turn`: the server lends the task sandbox to the
agent session through `sandbox_access`, and rows reach it through the `swe_gym` taskset. The `opencode_agent` README
still lists its session path as evaluation only, so check its token capture before you train on these rollouts.

`swe_gym_opencode_legacy_agent.yaml` is deprecated. It runs `opencode_sandboxed_agent` through the `legacy_agent`
relay, and that agent [will be removed](../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx)
after the Environment Server migration.

## Tests

```bash
gym env test --resources-server swe_gym
```

## Patch submission boundary

Candidate patches pass a strict intake gate before a verifier sandbox is allocated.
The limit is 1 MiB of UTF-8 text. Only canonical, unquoted ASCII paths and ordinary
regular-file text hunks are accepted. The gate rejects traversal, mismatched file
headers, duplicate sections, binary patches, renames/copies, symlinks, submodules,
and mode-only changes. New and deleted regular files are supported. This narrower
contract can reject previously accepted benchmark submissions; compare golden-patch
coverage before using it to produce benchmark results.

Edits to recognized test directories, `conftest.py`, pytest/build/environment
configuration, Python startup hooks, and known grader/request/result filenames
reject the **whole** submission. Paths in the trusted hidden-test patch are also
protected, including files outside conventionally named test directories. This
replaces the previous behavior of silently dropping candidate edits to hidden tests.
The list of recognized names is a conservative gate, not a complete inventory of
all files any project can use as grading inputs.

The fresh verifier checks its HEAD against the trusted task's `base_commit` and
checks every existing candidate path component with `lstat` before candidate code
or the repository install step runs. Symlinks, dangling symlinks, directories used
as file targets, and special files are refused. It applies the diff with `git apply
--check` and `git apply`; the fuzzy `patch` fallback is removed. These checks require
an audited task image with a trusted, clean working tree and no concurrent writer.
The preflight uses an isolated Python interpreter so repository startup modules do
not run during the check.

The existing two-sandbox lifecycle is retained. Only the patch is captured from the
agent; the verifier image, hidden tests, script, test lists, and grading authority
come from the server/task definition. No agent environment, test tree, request,
result file, or harness directory is copied into the verifier. The expected test
outcomes stay in the resources server. A small preflight JSON contains only the
trusted base commit and candidate paths.

Malformed/oversized submissions receive `reward=0`, `evaluation_completed=True`,
`mask_sample=False`, and `failure_kind=agent_invalid_output`. An unsafe path in the
fresh verifier is likewise an attributable failed submission. Capture failures,
wrong-base images, preflight failures, and provider errors are inconclusive and
masked with `failure_kind=verifier_error`. Invalid **golden** patches are masked as
verifier/data failures, rather than attributed to the policy. Provider-reported
interruptions and ambiguous exit codes 124/137 are masked; this environment cannot
reliably attribute them to the candidate. A nonzero test-process exit cannot earn
credit from passing-looking log lines.

Capture bounds the normal Git diff stream before provider transfer, disables Git
external diff/textconv, checks returned patch size again on the server, and retains
the existing capture deadline. The agent controls its executable/image state, so
this is not a provider-enforced output quota; malicious output producers still
need provider-side transport/output limits. `evaluation_timeout` must be positive
and finite. The verifier receives a fallback TTL when none is configured. CPU,
memory and disk ceilings remain the provider's responsibility; the shipped YAML
already requests these. When absent from a verifier config, the request defaults
to 2 CPUs, 4096 MiB memory and 8 GiB disk; null/nonpositive values are refused. Configure an isolated provider that enforces those limits.
Session ownership, bounded retries, stop-on-close/shutdown, and retained handles
when stop fails remain with the shared SWE sandbox-session implementation.

### Security scope and remaining grading gap

This is patch-boundary hardening, **not a claim of tamper-proof SWE verification**.
In particular, candidate source code is executable: it can influence pytest from
inside the test process, print forged test-status lines, or attempt to rewrite
verifier files at runtime under the existing shared user identity. Filtering test
filenames and creating a fresh sandbox cannot eliminate those attacks. The current
log parser remains a grading trust limitation even after these changes. The strict
expected-failure test `test_runtime_forged_success_remains_an_unclosed_grader_boundary`
records a passing-looking stdout payload that the current grader still credits.
It must stop being an expected failure when a stronger grading boundary is built.

A complete stronger boundary needs a task-specific test architecture: keep the
trusted controller and expected outputs outside candidate-controlled processes;
run candidate operations through a restricted, resource-bounded interface with
separate credentials/filesystem authority; compare observable outputs in the
trusted controller. A separate UID or read-only tests alone cannot prevent Python
code from monkeypatching assertions in its own test process. Do not describe this
SWE implementation as satisfying that stronger guarantee. Do not run it with the
`local` provider or with shared host mounts for untrusted submissions.

This change does not alter agent-image hygiene or outbound network access. Those
are separate controls. It also does not change search-only agents or their
server-side judges.

### Audit baseline and local checks

The implementation was audited against upstream NeMo Gym
`59d0b04f17cc5860e66bc51a35a7dcd8db0b1ebb`:

- `nemo_gym/verification/sandbox_verification.py` already creates one sandbox per
  call, delivers trusted files through `SandboxSpec.files`, uses a unique work
  directory, applies resources/TTL/deadlines, checks process/provider errors,
  validates the downloaded result, and attempts teardown in `finally`. It does not
  isolate the runner from code it launches or protect its result from a shared UID.
- `code_gen/sandbox_execution.py` and `competitive_coding_challenges/sandbox_evaluator.py`
  already use that helper. Code-gen removes its expected-output-bearing request
  file before execution; this does not make its in-process harness trustworthy
  against arbitrary introspection or mutation by submitted Python.
- SWE-Gym already uses a separate sandbox from the same task image and keeps hidden
  tests out of the agent session. Its shared SWE sandbox-session implementation
  owns retries and cleanup. Replacing that lifecycle with the generic helper would
  duplicate infrastructure and disrupt session retry/cleanup behavior.
- `terminal_multi_harness` grades tool-call action comparisons; it is not a SWE
  workspace patch verifier.

No provider implementation was added. No model/GPU rollout was run: these checks
exercise local trusted fixtures and mocked provider/session boundaries. Before
claiming rollout readiness, run representative model rollouts on a configured
isolated provider and inspect the agent, patch, verifier logs and outcome fields.

Local validation: 213 tests passed and one strict expected failure records the
unclosed stdout-forgery boundary. Scoped pre-commit/Ruff checks passed; the private
guardwords catalog was unavailable and its hook skipped that scan. Focused mypy
could not start because the locked development environment lacks `pathspec`.
No production-provider or model rollout claim is made.

Run the focused regression suite from the Gym checkout:

```bash
pytest resources_servers/swe_gym/tests \
  resources_servers/swebench/tests/test_patch_capture.py \
  tests/unit_tests/verification/test_sandbox_verification.py
```

The malicious patch fixtures cover traversal, conflicting headers, oversized
output, binary/special-file modes, grader/test/environment edits and trusted-tree
symlinks. A real temporary Git repository exercises an ordinary patch captured from
one workspace and applied to a separate trusted checkout. Session tests exercise
hidden-file separation, retries, failed starts/stops, replay, and final teardown.
