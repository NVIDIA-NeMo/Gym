# SaaS Bench evaluation

Two evaluation paths share the same 106 native benchmark tasks, fresh application
state, and native final-state verifier scripts:

- NeMo Gym → browser-use 0.13.7 → Eval Factory.
- NeMo Gym → OpenCode harness 2.1.1 (CLI 1.17.11) → Eval Factory.

The benchmark runtime, app definitions, container lifecycle/network helpers, native
prompt builder, both harness entry points, and all 106 verifier scripts ship in
`responses_api_agents/saas_bench_agent`. No separate SaaS Bench checkout is needed.
Task descriptions, metadata, input media, app images, and browser binaries remain
external data/binary assets. `prepare` verifies every task-data file against the
committed manifest, then joins the data and shipped code in a run-local bundle.
Reusing a bundle with changed code/data fails; use a fresh run directory.

## Select a harness

```bash
# Run inside the provisioned runtime image, from the pinned Gym checkout.
export SAAS_BENCH_DATA_DIR=/saas-data
export SAAS_BENCH_CACHE_ROOT=/cache/saas-bench
export SAAS_BENCH_WORKER_PYTHON=/opt/saas-worker/bin/python
export SAAS_BENCH_OPENCODE=/opt/saas/bin/opencode
export SAAS_NRT_ENROOT=1
export SAAS_ENROOT_IMAGE_DIR=/saas-images
export SAAS_ENROOT_RUNTIME_DIR=/saas-enroot-runtime
export PLAYWRIGHT_BROWSERS_PATH=/saas-browsers
# Select one: saas_bench_opencode or saas_bench_browser_use.
gym eval prepare --benchmark saas_bench_opencode
gym eval run --benchmark saas_bench_opencode --model-type vllm_model --split benchmark \
  +policy_base_url=http://MODEL_HOST:8000/v1 +policy_model_name=MODEL_NAME
```

The `saas_bench` benchmark name remains an OpenCode compatibility alias.
For a smoke, set `+saas_bench_tasks=<native-task-id>` in BOTH prepare and run;
prepare again when changing selectors; the two harnesses have separate prepared JSONLs.
Prepared files stay in the per-job Gym checkout under `benchmarks/<harness>/data/`;
Factory installs an isolated checkout for each job. Bundles and episodes use the run artifact directory.
Provide actor authentication with `SAAS_BENCH_MODEL_API_KEY` if the endpoint needs it.
Native judge-assisted verifier checks use `LLM_BASE_URL`, `LLM_API_KEY`, and
`LLM_JUDGE_MODEL` at runtime; no credentials are committed. Deterministic checks
keep their original state readers and weights. Do not replace them with a generic
trajectory judge.

## SVG assets and runtime

All paths below are beneath:
`/lustre/fsw/portfolios/nemotron/users/pritamb/gameplay_home/vlm_eval/saas_bench_assets/`.

| Asset | Relative SVG path | Container mount |
|---|---|---|
| Evaluation image | `serving_containers/saas-bench-eval-browser-use-0.13.7-opencode-1.17.11-svg-amd64.sqsh` | driver image |
| Task data | `artifacts/datasets/saas_bench_eval_task_data_31177a7/` | `/saas-data:ro` |
| App images (31 SQSH files) | `assets/enroot_images/` | `/saas-images:ro` |
| Rootless network and SquashFS helpers | `runtime/enroot/` | `/saas-enroot-runtime:ro` |
| Chromium and ffmpeg | `runtime/saas-bench-playwright-1.62.0-amd64/` | `/saas-browsers:ro` |

Bind a writable user-owned workspace cache to `/cache`, retain it in the mount
list, and create the bootstrap cache/tmp directories before launch. Rootless app
launch requires user/network/PID namespaces, Enroot, socat, slirp4netns, a static
BusyBox, and a C compiler for the UID shim. These are runtime prerequisites,
not packages to install into the public benchmark code. The shipped image builder
and pinned worker requirements are in `runtime-image/`; an image receipt records
its input/output hashes. Each task owns and tears down its own application state.
Use account `nemotron_n4_omni`; CPU allocations on SVG use partition `cpu`, QoS
`cpu-short`, with a four-hour limit. GPU-serving examples use `batch/normal`.
Keep all caches/artifacts outside `/root` and `/home/pritamb`.
Browser profiles and temporary data stay in the workspace cache. When its
path is too long for Unix sockets, a process-owned short `/tmp` symlink points
into that cache; the symlink is removed on exit. In containers, `/cache` itself
is already a short bind alias into the workspace.

## Results and failure semantics

Gym collects the original native weighted score (including fractional scores and
legitimate zeroes), PASS/FAIL, source/task/rollout bindings, verifier output, and
artifacts. Infrastructure and unavailable-verifier failures are masked and
reported separately. They are not counted as model-quality zeroes.
OpenCode preserves native tool events and screenshot timing. Browser-use preserves
its original browser transcript and screenshots; it is not relabelled as an
OpenCode trace. Saved evidence lives under the per-rollout `artifacts` directory.
No SFT/v5-generation stages or training changes are part of this evaluation adapter.

## Source and licensing

Runtime provenance: SaaS Bench commit
`31177a7a80040edf588388dd982af4b678d1e544`,
https://gitlab-master.nvidia.com/pritamb/saas-bench/-/tree/31177a7a80040edf588388dd982af4b678d1e544 .
The Apache-2.0 runtime is vendored with source headers, licenses, and SHA manifests.
Benchmark tasks/verifier scripts retain CC-BY-4.0 and attribution; see
`benchmark/NOTICE`, `benchmark/tasks/LICENSE`, and `benchmark/provenance.json`.
The vendored implementation is excluded from auto-fix linting to preserve source
semantics. Adapter, preparation, lifecycle, and protocol tests run on the actual
Gym response schemas; the benchmark is experimental until a full reference
score has been reproduced and reviewed.

The helper release also provides `ip`, `ss`, `unsquashfs`, `mksquashfs`, and libraries
missing from the base image. Runtime setup prepends its `usr/bin` and `usr/sbin`
to PATH and its library directory to LD_LIBRARY_PATH. Exact tool hashes are in
`runtime/enroot/saas-eval-tools.json`. To recreate these additions on SVG, run
`runtime-image/provision-helpers.py --destination <helper-release> --image-rootfs
<extracted-evaluation-image>`; it refuses to replace conflicting existing files.

### SVG writable app storage and judge transport

SVG deployments set `SAAS_ENROOT_PLATFORM_ROOT=/raid/enroot` and bind that platform
path into the driver. Extracted app root filesystems and live Enroot runtime use
its node-local `data/user-<SLURM_JOB_UID>/saas-<job>-<pid>` and `runtime/...` paths.
The Slurm UID is intentional: an inner Pyxis process can report UID 0. Keep image
layer caches, package caches, browser profiles, logs and artifacts in the mounted
workspace cache. Writable app root filesystems on the shared NFS workspace caused
permission errors and stale file handles; they are platform runtime, not caches.
Set `SAAS_STATIC_SHELL=/saas-enroot-runtime/usr/bin/busybox`. Runtime admission
rejects missing Enroot/network/UID helpers before dispatching episodes.

`SAAS_BENCH_JUDGE_HTTP_TIMEOUT=120` enables the Gym-owned native-verifier transport
wrapper. Only POST requests to the configured judge's `/chat/completions` endpoint
receive the longer timeout and one transient retry. Native verifier scripts,
criteria, weights and payloads remain unchanged; app requests keep their original
timeouts, and exhausted judge errors remain masked. The opted-in verifier process
budget is 600 seconds. With the option absent, the native launch path is unchanged.

SQLCMD exec additionally enters the target PID namespace and root filesystem so
its `/proc/self/exe`-relative resources can load. The ordinary namespace recipe
is retained for other executables. Explicit native `context load failed` and
`FATAL` diagnostics invalidate grading even if the task script emits zero-point
checks afterwards; they are not model-quality zeros. Native stderr is preserved
for diagnosis. Missing expected objects without a grader exception remain valid
failures with their original scores and weights.


### Failure outcomes and result compatibility

Judge HTTP exhaustion (including 429/5xx), invalid JSON and missing or empty
completions are recorded separately from task verifier text. This works for both
requests and urllib judge calls even when a task catches the exception and prints
FAIL. The entire row is masked with a retry/evaluation outcome; legitimate NO
answers remain task failures. Retry budgets remain owned by Gym/the caller.

Malformed model output, output caps, unrecovered context limits and actor timeouts
are explicit model failures counted as zero. Browser-use actor exceptions are
also model failures. Both harnesses still run final-state verification before
stopping applications, retaining measured checks for diagnostics. Verification
unavailability takes precedence over an actor failure. The typed
failure_outcome_action and failure_outcome_attribution fields survive JSON
serialization and recovery. Infrastructure startup failures remain retryable.

This corrects the scoring denominator and failure attribution. Existing artifacts
and active campaigns pinned to earlier Gym revisions are not rewritten; use a
fresh campaign at the corrected pin for comparable evaluation results.
