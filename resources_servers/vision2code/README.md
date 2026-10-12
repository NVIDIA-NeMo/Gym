# Vision2Code

A single-turn Gym environment for **ChemVQA-2K, DocVQA, ChartQA, and EEE-Bench**:
reference image → `simple_agent` Python submission → fresh Enroot render → vision rubric judge → reward.
It produces rollouts and rewards. RL trainer configuration is separate.

This environment adapts the scoring code and generation/judge prompts from
[image2code/vision2code](https://github.com/image2code/vision2code), by the
Vision2Code Authors, under Apache-2.0. It does **not** import or install the
upstream package. Gym owns model calls, orchestration, sandboxing, and artifacts.
[NOTICE](NOTICE), [LICENSE.upstream](LICENSE.upstream), and
[source provenance](tests/fixtures/provenance.json) record the source revision,
exact hashes, and local rubric modifications. The base upstream commit alone
does not reproduce the modified EEE background and whitespace scoring behavior.

This is a configurable environment using four Vision2Code rubrics, not a claim
to reproduce the complete upstream benchmark or its published results. Input
datasets retain their own licenses; the Apache-2.0 code license does not grant
rights to redistribute reference images. The committed example is synthetic.

The catalog entry is [environments/vision2code](../../environments/vision2code/manifest.yaml).
It declares a continuous reward in `[0, 1]` and stochastic judging. `verified: false`
and `experimental: true` remain set until broader baselining and maintainer review.

## Prepare tasks

Run commands from the Gym repository root, using Gym's Python environment with
`resources_servers/vision2code/requirements.txt` installed. A manifest is JSONL:

```json
{"id":"chem-001","image_path":"images/chem-001.png","rubric":"ChemVQA-2K"}
{"id":"doc-001","image_path":"images/doc-001.png","rubric":"DocVQA"}
{"id":"chart-001","image_path":"images/chart-001.png","rubric":"ChartQA"}
{"id":"eee-001","image_path":"images/eee-001.png","rubric":"EEE-Bench"}
```

Relative image paths resolve against the manifest directory. Use the exact rubric
names shown above; no aliases or other datasets are accepted.

```bash
python -m resources_servers.vision2code.prepare /data/manifest.jsonl /data/tasks.jsonl
```

Preparation validates each static PNG/JPEG/WebP/GIF, embeds its bytes in a
Responses `input_image` data URL, and records absolute reference path, dimensions,
and SHA-256 in `verifier_metadata`. It retains the upstream generation prompt and
adds explicit output dimensions, available libraries, and `/workspace/output.png`.
The verifier checks the reference hash again. References must be readable by the
resource server; they are never uploaded to the renderer.

For a portable first run, use `resources_servers/vision2code/data/example.jsonl`.
Its reference path resolves against the Gym checkout regardless of the resource
server's working directory. It contains an original synthetic chart released under Apache-2.0,
not a redistributed ChartQA example. `prepare.py` is the Gym row adapter; source
dataset downloading and split selection remain with the dataset owner.

## Build the renderer once

The [Dockerfile](container/Dockerfile) pins the Python base digest, Debian snapshot,
fonts, `unshare`, and Python dependencies. It includes matplotlib, NumPy, Pillow,
pandas, NetworkX, DejaVu, and Liberation fonts. It contains no inference libraries,
model weights, RDKit, Schemdraw, or notebook services.

On a Docker build host:

```bash
docker build -t vision2code-render:1 resources_servers/vision2code/container
enroot import -o /images/vision2code-render.sqsh dockerd://vision2code-render:1
```

Alternatively, build with Podman and import `podman://localhost/vision2code-render:1`.
On clusters without subordinate UID ranges, rootless Podman may need
`--storage-opt overlay.ignore_chown_errors=true`. The Dockerfile runs apt as the
build container's root to support this case. Use owned `XDG_RUNTIME_DIR` and
`ENROOT_TEMP_PATH` directories for image import. The resulting local `.sqsh` must
be accessible on the worker; evaluation does not import images or build containers.

## Check the target allocation

Run this inside the Slurm worker allocation **before evaluation**, with the same
Enroot installation and image. Point `--forbid-path` at an actual reference file
and any host credential directory that must be absent:

```bash
python -m resources_servers.vision2code.smoke \
  --image /images/vision2code-render.sqsh \
  --artifacts /results/vision2code/isolation \
  --forbid-path /data/images/chem-001.png
```

The probe checks blocked outbound networking, no network routes, an allowlisted
process environment, hidden parent processes, and absent forbidden paths. Each
render also checks namespace creation and mount visibility before executing code.
The host home mount is disabled and the rootfs is read-only. The bundled Enroot
[profile](enroot/mounts.d/10-renderer.fstab) selects only basic devices and private
shared memory, with no site/user hooks, host `/run`, or `/sys` mounts. This avoids
site hooks that would otherwise expose service-account credentials and Slurm IPC. Only the provider's unique `/workspace` mount is writable for task
files. Enroot remains an HPC execution boundary with limited isolation; this is
not a hardened adversarial multi-tenant runtime. Site hooks and mount configuration
must pass the actual allocation smoke. There is no unsandboxed fallback.

Slurm supplies the worker CPU/memory limits; Enroot does **not** enforce per-attempt
CPU/memory limits. Provision enough worker resources for four simultaneous renders,
or lower `render_concurrency`. Every attempt unpacks a fresh rootfs; inspect the
recorded startup time before considering any future reuse.

## Configure and evaluate

Both adapters connect to existing OpenAI-compatible chat-completions endpoints;
this configuration does not launch inference services. Both models must
accept image inputs. Policy and judge may use independent hosts, models, and keys.
The endpoint URLs should include `/v1`.

```bash
export VISION2CODE_SQSH=/images/vision2code-render.sqsh
export VISION2CODE_ARTIFACTS=/results/vision2code/artifacts
export VISION2CODE_POLICY_URL=http://policy-host:8000/v1
export VISION2CODE_POLICY_MODEL=your-served-vision-model
export VISION2CODE_JUDGE_URL=http://judge-host:8000/v1
export VISION2CODE_JUDGE_MODEL=your-served-vision-judge
# VISION2CODE_POLICY_KEY and VISION2CODE_JUDGE_KEY default independently to EMPTY.

gym env start --config resources_servers/vision2code/configs/vision2code.yaml

gym eval run --no-serve --agent vision2code_simple_agent \
  --input /data/tasks.jsonl --output /results/vision2code/rollouts.jsonl \
  --num-repeats 1 --temperature 0.7 --max-output-tokens 8192
```

`simple_agent` has no tools and `max_steps: 1`. No repair turn or judge feedback is
sent to the policy. The resource config controls judge reference/sampling, local
image/provider storage paths, execution timeout (60 seconds), render concurrency
(4), and artifact directory. Namespace/mount behavior cannot be weakened through
provider overrides. Judge transport and its retries are Gym's `call_judge`.

The optional [hosted Qwen config](configs/vision2code_qwen.yaml) inherits this
composition and overrides only policy/judge model settings. Set `NVIDIA_API_KEY`
and `VISION2CODE_SQSH`, then pass that file to `gym env start --config`. Both roles
use the same model in this convenience config; self-judged smoke rewards are not
an independent benchmark quality estimate. For reasoning models, set the model
adapter's `uses_reasoning_parser` according to the endpoint's response format.

For a smoke test, limit collection to one row with `--limit 1`. Inspect its Python,
rendered PNG, rubric selection, reward, cleanup diagnostics, and timings. Broader
evaluation should cover each rubric with independently selected policy and judge models.

## Results and artifacts

`POST /verify` takes the normal Gym response and task row, and returns `reward`,
`render`, `rubric_result`, `judge_diagnostics`, `artifacts`, and `timings` alongside
the normal Gym fields. Reward is `final_rating_0_to_5 / 5.0`; the upstream category
weights, decimal rounding, critical-category caps, EEE background treatment, and
blank-image inspection are preserved. Sparse whitespace is diagnostic only.

| Outcome | Reward | Masked | Judge called |
| --- | --- | --- | --- |
| Valid PNG | rubric rating / 5 | no | yes |
| Valid blank PNG | existing rubric blank cap applies | no | yes |
| Code error, timeout, missing/invalid PNG | 0 | no | no |
| Sandbox/transfer/isolation failure | 0 placeholder | yes | no |
| Judge transport or malformed/invalid score | 0 placeholder | yes | attempted |
| Missing/changed reference | 0 placeholder | yes | no |

Failures use `mask_sample`, `failure_kind`, `failure_reason`, compatibility
`instance_config.mask_sample`, and `_ng_failure_class` so Gym excludes them from
quality aggregates and routes them to its failure sidecar. A malformed judge reply
is deliberately masked in this environment, overriding Gym's usual wrong-answer
interpretation. There is no judge repair conversation or custom retry loop.

Each attempt has a UUID directory containing:

- `code.py`: normalized raw/fenced submission, without code repair.
- `output.png`: only when a valid PNG was downloaded before sandbox teardown.
- `execution.json`: process results, isolation/mount checks, dimensions, cleanup,
  and startup/render/cleanup/total seconds.
- `judge.json`: received judge response and candidate inspection, or transport error.
- `result.json`: reward, mask, rubric category breakdown/caps, diagnostics, artifact
  paths, and timings; omits the large input data URLs and policy response.

The renderer rejects symlinks, nonregular files, non-PNG/unreadable images,
animated PNGs, files over 32 MiB, and images over 25 million pixels. Dimension
mismatches are recorded without resizing or rewriting the output. `savefig` is
never overridden. Cancellation waits for bounded start/exec operations and sandbox
stop before propagating; `execution.json` records cleanup failures if infrastructure
prevents teardown. A hard worker kill cannot run Python cleanup.

## Tests

```bash
python -m pytest resources_servers/vision2code/tests

# With the configuration variables above set (no inference or containers started):
gym env validate vision2code
gym env test vision2code
gym env publish vision2code

# Real execution, inside the intended worker allocation:
VISION2CODE_TEST_SQSH=/images/vision2code-render.sqsh \
  python -m pytest resources_servers/vision2code/tests/test_enroot_integration.py

pre-commit run --files $(git ls-files --cached --others --exclude-standard resources_servers/vision2code)
```

The committed golden fixtures were produced from the existing upstream checkout,
not from the port, and require no external repository. They cover all four rubric
definitions and prompts, scores/rounding/caps, sparse/blank handling, and invalid
scores. Tests also cover preparation, Gym image schemas, failure masking,
concurrency, transfers, limits, dimensions, and cancellation during startup and exec.

The exported verifier fixture exercises production rubric aggregation and reward
normalization for full, zero, and malformed scores across all four rubrics. It
does not attest live judge quality or sandbox behavior; those have separate server,
Enroot, and model rollout checks. The current legacy CI data validator still asks
for five example rollouts even for manifest workloads; this environment follows
the manifest contract and does not fabricate extra rollouts to satisfy that check.
