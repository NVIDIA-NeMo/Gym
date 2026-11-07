# Native SOL-ExecBench verifier

This resources server runs the public [SOL-ExecBench evaluator](https://github.com/NVIDIA/SOL-ExecBench)
at revision `a9fa0804c793d438e70850c33fe34426e66d53dd` in a fresh OpenSandbox sandbox with one GPU per response.
It accepts native Solution JSON with inline source files, uses server-owned problem definitions and workloads,
and returns native correctness, GPU timings, and an anchored **SOLscore reward averaged over every workload**.
Incorrect workloads contribute zero; full-problem correctness is reported separately as `solved`. Scoring requires a
reviewed, UUID-keyed anchor manifest. Current reference timing is not a scoring baseline or SOL anchor.

This integration is experimental. Mocked transport and synthetic result tests do not establish a working GPU image,
OpenSandbox deployment, or model rollout. These must be qualified before reporting benchmark results.

## Inputs and configuration

Prepare data with [`benchmarks/solexecbench`](../../benchmarks/solexecbench/README.md).
The model returns a native Solution JSON object (optionally in one `json` fence), including `name`, `definition`,
`author`, `spec`, and nonempty `sources` with relative `path` and inline `content`.
The native language/build schema is validated inside the pinned image. Native hardware labels are `B200` and `LOCAL`;
`H100` is not a native solution-schema hardware label. The public benchmark uses B200.

Rows supply only `verifier_metadata: {task_id, problem_digest}`. The server loads
`problem_manifest_path` and verifies its exact bytes against `problem_manifest_sha256`.
Each trusted problem contains the inline native definition, all workloads, and SHA256-pinned relative asset paths.
Definition dictionary order is preserved because it defines the argument ABI. Asset paths must remain inside the
manifest directory, including after symlink resolution. Assets are checked before upload. Corpus files are downloaded
at runtime and are not included in this repository.

Compose `configs/sol_execbench.yaml`, the benchmark config, an existing model config, and
`nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`. Supply:

- `problem_manifest_path` and the bare SHA256 from the preparation sidecar;
- `anchor_manifest_path` and its exact-byte `anchor_manifest_sha256`;
- an absolute writable `artifact_root`;
- `sandbox_image` in immutable `registry/image@sha256:...` form;
- a configured OpenSandbox endpoint, credentials, and one-GPU-capable deployment;
- `samples_per_task` matching the evaluation repeat count.

OpenSandbox command retries must remain zero. Each resources worker serializes its evaluations and each evaluation
creates one disposable sandbox, so no two evaluators share a sandbox. The sandbox image entrypoint is overridden with
an idle process. One visible GPU and the exact `NVIDIA B200` product name are checked before running the candidate;
GB200 and MIG device names are rejected. The deployment must provide exclusive GPU allocation; visible-device count
and process observations alone do not prove exclusivity.
`LOCAL` is available for explicit noncanonical development runs and changes protocol identity.

The native defaults are explicit: compilation 120 seconds, evaluation 600 seconds, 10 warmups, 50 timing iterations,
seed 200, `lock_clocks=false`, `benchmark_reference=false`. The outer command watchdog defaults to 900 seconds and
must exceed compilation plus evaluation budgets. Changing these settings changes the recorded protocol digest.
Enable `benchmark_reference` explicitly to expose current native reference GPU timings; some references are expensive.
RPC or wall time is never used as kernel latency.

## SOLscore and anchors

For a correct workload, the [pinned scoring formula](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/sol_score.py)
is `1 / (1 + (Tk - Tsol) / (Tb - Tsol))`, with all times in milliseconds. `Tk` is the measured candidate latency,
`Tb` the reviewed optimized baseline, and `Tsol` the hardware speed-of-light bound. The baseline scores 0.5 and a
candidate at the SOL bound scores 1. Proven candidate-failure workloads score zero. The response's `sol_score` and
Gym `reward` follow the [current website](https://research.nvidia.com/benchmarks/sol-execbench/blog/introducing-sol-execbench):
the arithmetic mean over all workloads, including those zeros. A partly correct solution can earn credit while
`solved` remains false. Malformed Solution JSON receives zero.

The [paper's scoring domain](https://arxiv.org/html/2603.19173v1#S4.SS3) requires `Tb > Tsol` and `Tk >= Tsol`.
Invalid anchors fail startup. A measured candidate below its SOL bound requires an audit: the sample is masked,
`sol_score` is null, and available correctness and timing diagnostics remain visible. Scores are not clipped or
replaced with reference speedups.

The server verifies the anchor file's SHA256, selected problem-manifest SHA256, target hardware, every problem
digest, and exact task/UUID coverage before GPU work. Duplicate JSON keys are rejected. Each workload entry contains
`baseline_ms` and `sol_ms`; the manifest records `provenance.source` and `provenance.revision`. See
[`AnchorManifest`](scoring.py) for the complete schema. Anchor identity, the scoring rule, and native benchmark
configuration are included in the recorded protocol.

No verified public UUID-keyed anchor export is supplied. The pinned dataset has no timing anchors, and published
timing tables omit workload UUIDs. Three tasks contain duplicate-axis workloads, so axes alone cannot establish
anchor identity. Obtain a reviewed export covering every selected UUID; this integration does not guess mappings
or substitute live reference timings.

## Runtime image

Build the upstream image from the pinned public source, then extend it with the revision marker:

```bash
git clone https://github.com/NVIDIA/SOL-ExecBench.git /tmp/sol-execbench-native
git -C /tmp/sol-execbench-native checkout a9fa0804c793d438e70850c33fe34426e66d53dd
docker build --platform linux/amd64 \
  --build-arg HOST_UID=1001 --build-arg HOST_GID=1001 \
  -f /tmp/sol-execbench-native/docker/Dockerfile \
  -t sol-execbench-native:a9fa080 /tmp/sol-execbench-native
docker build --platform linux/amd64 \
  --build-arg SOL_EXECBENCH_BASE_IMAGE=sol-execbench-native:a9fa080 \
  -f resources_servers/sol_execbench/Dockerfile -t REGISTRY/solexecbench-gym:qualification .
```

The explicit platform supports GPU hosts when building from an ARM development machine. UID/GID 1001 avoid the
Ubuntu base image's existing UID 1000; these are supported upstream build arguments. With a Buildx builder that
does not load images into the local Docker image store automatically, use `--load` and ensure the second build can
resolve the first image.

Publish to an operator-controlled registry and configure the resulting **registry manifest digest** after
qualification.
No ready-to-use image digest is provided here. The runner checks both `/opt/sol-execbench-revision` and every installed
native package Python source against `native_source_hashes.json`; a matching marker alone is insufficient.
Hashes are metadata from the pinned Apache-2.0 evaluator, not vendored source.

## Failure and evidence contract

Full-problem correctness requires native-schema-validated traces bound to each trusted workload payload exactly once,
matching definition and solution identities, all `PASSED`, and the native CLI's success exit code. Complete
candidate-failure traces are accepted with exit code 1. Incorrect shape, dtype, numerical results and native
compile/reward-hack statuses identify failed workloads. Native CLI compile failures that produce no traces remain
unresolved.

Timeouts, missing/partial traces, invalid references, unexpected exits, transport errors, and cleanup failures set
`mask_sample=true`, with an explicit failure kind and null SOL score. The native `RUNTIME_ERROR` status is ambiguous
(it also covers missing inputs, clock failures and timing failures), so it remains masked. Available native traces,
workload statuses, correctness diagnostics, and GPU timings are retained. The required numeric Gym reward is zero
on masked responses; that placeholder must not be used as a candidate-failure measurement.

Aggregate `correctness_complete` requires every declared task/repeat slot and no infrastructure failures.
`score_complete` additionally requires a valid score for every slot. Thus an otherwise complete native pass awaiting
SOL-bound review can retain correctness metrics while score metrics remain null. `correctness_at_1` and `pass_at_N`
report correctness; `sol_score` averages each task's sample scores and then averages tasks equally.
`sol_score_best_of_N` separately averages each task's best sample score. No missing or masked score is replaced by zero,
and these metrics do not imply reproduction of the published evaluation protocol.

Every attempt records the request, protocol, candidate, native input/config files, runner stdout/stderr, native
stdout/stderr, raw traces, GPU identity, memory/power/clock observations, compute-process observations and result under
`artifact_root/<request_sha256>/`. Native source hashes,
image digest, trusted problem and anchor manifest digests, runner digest, scoring rule, and native configuration
define protocol identity.
Concurrent identical requests share the same attempt; completed results replay without GPU work. An existing incomplete
attempt stays unresolved rather than silently rerunning. Sandboxes are stopped in `finally`, including on cancellation.

Isolation protects the Gym host and separates candidates. The native evaluator imports candidate code in its worker
and owns its correctness/timing defenses. This adapter does not establish a new adversarial oracle boundary or validate
every timed output independently of the native evaluator.

## Validation and license

Run `pytest resources_servers/sol_execbench/tests` and the environment's verifier fixture in Gym's dependency
environment.
The fixture tests production scoring with original synthetic traces and anchors; transport tests mock the OpenSandbox
API. Neither claims GPU execution. A representative real model rollout remains required before readiness.

This integration is Apache-2.0. The evaluator is Apache-2.0. Public dataset terms are separate: download and review its
license at runtime. Its evaluation-only terms prohibit training and redistribution; see the
[dataset license discussion](../../benchmarks/solexecbench/README.md#licensing). No corpus content or private subset
is redistributed in these fixtures.
