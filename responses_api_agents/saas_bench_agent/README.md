# SaaS Bench evaluation

Two evaluation paths share the same 106 native benchmark tasks, fresh application
state, and native final-state verifier scripts:

- NeMo Gym → browser-use 0.13.7 → Eval Factory.
- NeMo Gym → OpenCode harness 2.1.1 (CLI 1.17.11) → Eval Factory.

The benchmark runtime, app definitions, container lifecycle/network helpers, native
prompt builder, both harness entry points, and 11 modified verifier scripts ship in
`responses_api_agents/saas_bench_agent`. No separate SaaS Bench checkout is needed.
The 95 unmodified upstream verifiers, task descriptions, metadata, input media, app images, and browser binaries remain
external data/binary assets. `prepare` verifies every task-data file against the
committed manifest, then joins the data and shipped code in a run-local bundle.
Reusing a bundle with changed code/data fails; use a fresh run directory.

## Select a harness

```bash
# Run inside the provisioned runtime image, from the pinned Gym checkout.
export SAAS_BENCH_DATA_DIR=/saas-data
export LLM_BASE_URL=https://YOUR_JUDGE_HOST/v1
export LLM_JUDGE_MODEL=YOUR_JUDGE_MODEL
# Supply LLM_API_KEY securely at runtime.
export SAAS_BENCH_CACHE_ROOT=/cache/saas-bench
export SAAS_BENCH_WORKER_PYTHON=/opt/saas-worker/bin/python
export SAAS_BENCH_OPENCODE=/opt/saas/bin/opencode
export SAAS_NRT_ENROOT=1
export SAAS_ENROOT_IMAGE_DIR=/saas-images
export SAAS_ENROOT_RUNTIME_DIR=/saas-enroot-runtime
export PLAYWRIGHT_BROWSERS_PATH=/saas-browsers
# Select one: saas_bench_opencode or saas_bench_browser_use.
gym eval prepare --benchmark saas_bench_opencode ++saas_bench_run_root=/runs/saas-kimi-001
gym eval run --benchmark saas_bench_opencode --model-type vllm_model --split benchmark ++saas_bench_run_root=/runs/saas-kimi-001 \
  +policy_base_url=http://MODEL_HOST:8000/v1 +policy_model_name=MODEL_NAME
```

The two supported benchmark names are `saas_bench_opencode` and `saas_bench_browser_use`; the unused `saas_bench` alias has been removed.
Every invocation requires an explicit, unique `saas_bench_run_root`. Reusing a run root resumes that run and reuses saved scores; use a fresh root for each model/checkpoint. Dispatch concurrency defaults to the agent concurrency so queued rows do not spend the episode deadline waiting for a slot.
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

Choose an asset root accessible to your deployment. The relative paths below
identify the required releases; bind them to the listed container paths.
Site-specific filesystem paths and Slurm settings belong in deployment configuration.

| Asset | Relative asset path | Container mount |
|---|---|---|
| Evaluation image | `serving_containers/saas-bench-eval-browser-use-0.13.7-opencode-1.17.11-svg-amd64.sqsh` | driver image |
| Task data | a checkout of UniPat-AI/SaaS-Bench at `48c22deb` or a release containing all 323 manifest files | `/saas-data:ro` |
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
Use the account, partition, QoS and walltime approved for your deployment.
Keep caches and artifacts in the mounted workspace rather than user homes.
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

Upstream: https://github.com/UniPat-AI/SaaS-Bench at
`48c22deb18b98c0ed78f81e7f3be82bc162de2c8` (before v1.1).
Runtime code: Apache-2.0. Task/graders/media: CC-BY-4.0, retaining author notices.
See `benchmark/VENDORING.md` for source mappings and every modified grader.
Eleven graders differ from that public release: seven pre-existing semantic/schema
adaptations plus four Drupal state-reader fixes. In particular, `business_052`
requires exactly 24 percent instead of upstream's approximate 10.04 or 24 percent.
Scores are therefore not claimed equivalent to unmodified public upstream.
The 95 unmodified graders are external, pinned byte-for-byte in the data manifest;
prepare and startup both validate their bytes. NVIDIA-authored extensions are
identified separately in provenance, rather than attributed to upstream authors.
The benchmark remains experimental pending full reference qualification.

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

`SAAS_BENCH_JUDGE_HTTP_TIMEOUT=120` configures the Gym-owned native-verifier transport
wrapper. Only POST requests to the configured judge's `/chat/completions` endpoint
receive the longer timeout and one transient retry. Verifier prompts and payloads are retained; the 11 disclosed grader adaptations still apply; app requests keep their original
timeouts, and exhausted judge errors remain masked. The verifier process
budget is 600 seconds. The fault channel is always active; this option controls the HTTP timeout.

SQLCMD exec additionally enters the target PID namespace and root filesystem so
its `/proc/self/exe`-relative resources can load. PHP reads use a stable target-root working directory; PostgreSQL retains its
existing namespace recipe. Explicit native `context load failed` and
`FATAL` diagnostics invalidate grading even if the task script emits zero-point
checks afterwards; they are not model-quality zeros. Native stderr is preserved
for diagnosis. Missing expected objects without a grader exception remain valid
failures with their original scores and weights.


### Failure outcomes and result compatibility

Judge HTTP exhaustion (including 429/5xx), invalid JSON and missing or empty
completions are recorded separately from task verifier text. This works for both
requests and urllib judge calls even when a task catches the exception and prints
FAIL. The entire row is masked with a retry/evaluation outcome; legitimate NO
answers remain task failures. Verification retries occur inside the episode before teardown: `+saas_bench_inconclusive_verification_retries=1` permits one retry (range 0–3), with a separate retained receipt per attempt. Exhausted judge/verifier faults remain masked with terminal `fail`/evaluation attribution, because detached reverify cannot reconstruct live app state.

Confirmed malformed model output, output caps and unrecovered model context
limits retain their measured final-state grade (including partial credit). Browser-use preserves explicit model-call failure metadata;
unknown actor exceptions, Chrome/CDP startup, browser observation/screenshot faults
and provider failures stay retryable evaluation faults, including after partial
progress. An OpenCode timeout is tagged `agent_timeout` only after generated
content, an action or output-token evidence establishes a model turn. Empty or
startup-only timeout traces remain retryable evaluation faults. Both harnesses run final-state verification for confirmed model endings before
stopping applications, using measured checks as the reward. A verifier missing its final SCORE line or a task app failing readiness makes grading inconclusive, even if earlier checks passed. Verification
unavailability takes precedence over an actor failure. The typed
failure_outcome_action and failure_outcome_attribution fields survive JSON
serialization and recovery. Infrastructure startup failures remain retryable.

This corrects the scoring denominator and failure attribution. Existing artifacts
and active campaigns pinned to earlier Gym revisions are not rewritten; use a
fresh campaign at the corrected pin for comparable evaluation results.


Public code provenance links resolve to the initial Gym import snapshot above.
Original source revisions, original-file hashes, author notices and Apache-2.0
licenses remain attested in `benchmark/provenance.json`; no private checkout is
required to inspect or run the shipped code. Header-link cleanup does not change
any Python AST or task check. Dataset preparation binds to the shipped file
bytes, so prepare a fresh bundle after changing revisions.

### Current harness scope

This uses legacy_agent and a SaaS-specific OpenCode worker, not stock opencode_agent. It does not promise cross-harness score parity: protocol repair and browser tooling differ. The recommended resources-server + stock-harness migration is documented in `benchmark/VENDORING.md` as follow-up work. Both requested harnesses remain supported here.

Enroot app networks disable host-loopback access. Docker-compatible exec uses only host PATH and explicit `-e`/runtime flags; it does not pass judge keys or Gym configuration into app commands. Cancellation reaps detached Enroot app processes even during readiness startup. Deployment platform roots are explicit absolute paths, rather than a single cluster-specific hardcoded root.
