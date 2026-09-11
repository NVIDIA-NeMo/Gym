# Scale-SWE resources server

Verification for [AweAI-Team/Scale-SWE](https://huggingface.co/datasets/AweAI-Team/Scale-SWE):
20,181 software-engineering tasks, entirely Python.

Unlike the SWE-rebench server, there is exactly one test runner here (pytest), so there is no
per-row log-parser dispatch. Each row ships its own prebuilt image (`aweaiteam/scaleswe:...`),
an explicit `workdir`, and a `pre_commands` string that performs the checkout of
`parent_commit` and scrubs the repo's git history so the fix commit is unreachable.

Sandboxes come from `nemo_gym.sandbox`, so the same server runs on OpenSandbox or any other
configured provider.

## The failing tests

A row supplies the tests that must go from failing to passing one of two ways, and 61% of rows
carry both:

- `f2p_script` (92% of rows): a literal pytest file, uploaded to `/tmp` and copied into the
  checkout as `test_fail_to_pass.py` (the name the dataset's own `FAIL_TO_PASS` ids expect). It
  cannot be written into the repo before `pre_commands` runs, because `pre_commands` includes
  `git clean -fd`, which would delete it.
- `f2p_patch` (69% of rows): a diff that adds the tests directly into the existing test tree.

pytest is pointed at the **unique test files** named by `FAIL_TO_PASS ∪ PASS_TO_PASS`, not at
every individual node id — a row can carry hundreds of ids across up to ~69 files, and grading
by id afterwards from the collected output gives the same verdict without risking a pathological
command line.

## Grading

An instance resolves only when **every** `FAIL_TO_PASS` and `PASS_TO_PASS` node id is observed
as `PASSED` in the pytest output. A node id absent from the output counts as not passing —
treating absent as success is the standard way a broken test command (or a collection error)
scores as a resolved instance.

`FAIL_TO_PASS` / `PASS_TO_PASS` arrive as JSON-encoded strings in this dataset, not lists; see
`as_id_list`.

The graded region of the log is delimited by markers, so setup-time noise (pip installs,
`pre_commands` output) never reaches the pytest-output parser.

## Prepare the data

```bash
# The whole set, or a bounded trial:
SCALE_SWE_LIMIT=200 python benchmarks/scale_swe/prepare.py
```

`prepare.py` writes the full 20,181-row set, unfiltered. `data/supported_instance_ids.txt` is a
separate artifact for training use: the 17,696 instances (of 20,181) whose golden patch resolved
in all three passes of a full-set 3x sweep (see Golden-patch validation below). The other 2,485
are excluded from that list because none can be scored either way: 1,671 never resolve, 709 never
produced a verdict (image pull / setup failures, not row failures), and 105 resolve
inconsistently across passes.

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.
A row whose golden patch does not resolve cannot be used for evaluation or training.

```bash
gym env start \
  --config resources_servers/scale_swe/configs/scale_swe.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/scale_swe/apply_golden_patch.py \
  +benchmark_jsonl=benchmarks/scale_swe/data/scale_swe_benchmark.jsonl \
  +limit=40 +concurrency=8
```

It prints a resolved rate overall and per language (this set is 100% Python, so that breakdown
is a single row), and writes one row per task. `diagnose_failures.py` separates environment
failures (OOM, disk, network, timeout, no-verdict, zero-tests-collected) from genuine row
failures — see `resources_servers/swe_rebench/diagnose_failures.py` for the shape of evidence
this looks for and why each cause is only reported on direct signal in the captured output, not
inferred from the fact that a task failed.

`aggregate_golden_patch.py` joins repeated golden-patch passes (e.g. a 3x sweep) into
supported / flaky / broken / inconclusive buckets, keeping an infra fault (no verdict) separate
from a genuinely nondeterministic test so neither silently corrupts the other's label.

## Running an agent

`benchmarks/scale_swe/opencode.yaml` wires this server to the opencode sandboxed agent, in the
same shape as the swebench verified/pro/multilingual and swe_rebench benchmarks.
