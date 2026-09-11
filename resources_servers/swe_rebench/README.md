# SWE-rebench-V2 resources server

Verification for [nebius/SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2):
32,079 software-engineering tasks across 20 languages.

Unlike the SWE-bench servers, there is no image repository or per-language configuration here.
Every row ships its own prebuilt image (`docker.io/swerebenchv2/...`), the commands to install
and to test it, and the name of the upstream parser for its test log. Verification therefore
reduces to: start the row's image, apply the patches, run the row's test command, parse, grade.

Sandboxes come from `nemo_gym.sandbox`, so the same server runs on OpenSandbox or any other
configured provider.

## Grading

An instance resolves only when **every** `FAIL_TO_PASS` and `PASS_TO_PASS` test is observed and
passing. A test that is missing from the parsed output counts as not passing — treating absent
as success is the standard way a broken test command scores as a resolved instance.

Test names are normalised to strip per-run timings (`[123 ms]`, `(1.5 s)`, `in 2.0 sec`) before
comparison, matching the existing `swe_agents` SWE-rebench grader so the two agree.

The graded region of the log is delimited by markers, so install-time noise never reaches the
parser. Several parsers are line-oriented and would otherwise read dependency-resolver output
as test results.

## Log parsers

The dataset names one of 34 upstream parsers per row. Rather than reimplement that set, the
MIT-licensed `log_parsers.py` from
[SWE-rebench/SWE-rebench-V2](https://github.com/SWE-rebench/SWE-rebench-V2) is fetched on first
use and called directly; see `log_parsers.py`. An unknown parser name is an error, never a
fallback: a wrong parser returns an empty result, which grades as "nothing passed" and is
indistinguishable from a real failure.

## Prepare the data

```bash
# The whole set, or a subset:
SWE_REBENCH_LANGUAGES=python,go SWE_REBENCH_LIMIT=200 \
  python benchmarks/swe_rebench/prepare.py
```

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.
A row whose golden patch does not resolve cannot be used for evaluation or training.

```bash
gym env start \
  --config resources_servers/swe_rebench/configs/swe_rebench.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/swe_rebench/apply_golden_patch.py \
  +benchmark_jsonl=benchmarks/swe_rebench/data/swe_rebench_benchmark.jsonl \
  +limit=40 +concurrency=8
```

It prints a resolved rate overall and per language, and writes one row per task.

On a Slurm cluster where OpenSandbox is only reachable from inside, use
`temp_launch_swe_rebench_golden_patch.sh`, which runs the same thing on the **cpu** partition —
the workload needs no GPU, since the containers run remotely.

`aggregate_golden_patch.py` joins repeated golden-patch passes (e.g. a 3x sweep) into
supported / flaky / broken / inconclusive buckets, keeping an infra fault (no verdict) separate
from a genuinely nondeterministic test so neither silently corrupts the other's label.
`data/supported_instance_ids.txt` is the result of one such sweep over the full set: 22,684 of
32,079 instances whose golden patch resolved in all three passes -- a separate
artifact for training use, not consumed by `prepare.py` or the eval benchmark. The other 9,395
are excluded from that list because none can be scored either way: 6,813 never resolve
(concentrated in the compiled-language rows -- java, kotlin, scala, cpp), 2,376 resolve
inconsistently across passes (flaky tests), and 206 never produced a verdict in any pass (image
pull / setup failures, not row failures -- see `diagnose_failures.py`).

## Running an agent

`benchmarks/swe_rebench/opencode.yaml` wires this server to the opencode sandboxed agent, in the
same shape as the swebench verified/pro/multilingual benchmarks.

Upstream SWE-rebench code is MIT licensed. This adapter is Apache-2.0.
