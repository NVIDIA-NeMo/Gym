# Native SOL-ExecBench verifier

This resources server runs the public [SOL-ExecBench evaluator](https://github.com/NVIDIA/SOL-ExecBench)
at revision `a9fa0804c793d438e70850c33fe34426e66d53dd` in a fresh OpenSandbox sandbox with one GPU per response.
It accepts native Solution JSON with inline source files, uses server-owned problem definitions and workloads,
and records native correctness traces and GPU timing. The Gym reward is **all-workloads correctness (0 or 1)**.
`sol_score` is null: current reference timing is not a published SOL anchor, and this integration does not infer
anchors.

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

## Runtime image

Build the upstream image from the pinned public source, then extend it with the revision marker:

```bash
git clone https://github.com/NVIDIA/SOL-ExecBench.git /tmp/sol-execbench-native
git -C /tmp/sol-execbench-native checkout a9fa0804c793d438e70850c33fe34426e66d53dd
docker build -f /tmp/sol-execbench-native/docker/Dockerfile \
  -t sol-execbench-native:a9fa080 /tmp/sol-execbench-native
docker build --build-arg SOL_EXECBENCH_BASE_IMAGE=sol-execbench-native:a9fa080 \
  -f resources_servers/sol_execbench/Dockerfile -t REGISTRY/solexecbench-gym:qualification .
```

Publish to an operator-controlled registry and configure the resulting **registry manifest digest** after
qualification.
No ready-to-use image digest is provided here. The runner checks both `/opt/sol-execbench-revision` and every installed
native package Python source against `native_source_hashes.json`; a matching marker alone is insufficient.
Hashes are metadata from the pinned Apache-2.0 evaluator, not vendored source.

## Failure and evidence contract

A pass requires native-schema-validated traces with each trusted workload UUID exactly once, matching definition and
solution identities, all `PASSED`, and the native CLI's success exit code. Complete candidate-failure traces are
accepted
with exit code 1. Incorrect shape, dtype, numerical results and native compile/reward-hack statuses receive zero
correctness.
Malformed Solution JSON also receives zero. Native CLI compile failures that produce no traces remain unresolved.

Timeouts, missing/partial traces, invalid references, unexpected exits, transport errors, and cleanup failures set
`mask_sample=true`, with an explicit failure kind and null SOL score. The native `RUNTIME_ERROR` status is ambiguous
(it also covers missing inputs, clock failures and timing failures), so it remains masked. The required numeric Gym
reward field is zero on masked responses; it is not a candidate-failure measurement. Official aggregate metrics are
null
until every declared task/repeat slot is present and measured. No timeout-zero sensitivity score is reported.

Every attempt records the request, protocol, candidate, native input/config files, runner stdout/stderr, native
stdout/stderr, raw traces, GPU identity, memory/power/clock observations, compute-process observations and result under
`artifact_root/<request_sha256>/`. Native source hashes,
image digest, trusted manifest digest, runner digest, and native configuration define protocol identity.
Concurrent identical requests share the same attempt; completed results replay without GPU work. An existing incomplete
attempt stays unresolved rather than silently rerunning. Sandboxes are stopped in `finally`, including on cancellation.

Isolation protects the Gym host and separates candidates. The native evaluator imports candidate code in its worker
and owns its correctness/timing defenses. This adapter does not establish a new adversarial oracle boundary or validate
every timed output independently of the native evaluator.

## Validation and license

Run `pytest resources_servers/sol_execbench/tests` and the environment's verifier fixture in Gym's dependency
environment.
The fixture tests the production native-result classifier with original synthetic traces; transport tests mock the
OpenSandbox API. Neither claims GPU execution. A representative real model rollout remains required before readiness.

This integration is Apache-2.0. The evaluator is Apache-2.0. Public dataset terms are separate: download and review its
license at runtime; no evaluation-only corpus content or private subset is redistributed in these fixtures.
