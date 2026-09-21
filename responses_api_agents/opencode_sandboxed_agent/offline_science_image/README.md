# Offline scientific base image

This image supplies OpenCode and offline scientific tools. Scientific Python and
Sage both use Python 3.13.14, with separate dependency locks. The dedicated agent
runs Gym on the host; no Gym installation or source upload is needed inside the
sandbox. Pi adds its CLI in a [small runtime layer](../../../fern/versions/latest/pages/evaluation/harness.mdx#reproducible-offline-images).

Build from this directory:

```sh
docker build --platform linux/amd64 -t <registry/repository>:<tag> .
```

The image contains three independent toolchains:

- **Python 3.13.14** with the scientific packages in `requirements.in`. Exact
  versions and artifact hashes are in `requirements.lock`; PyTorch uses CPU wheels.
- **SageMath 10.8**, including GAP, PARI and Singular, in `/opt/sage`. Its own
  Python/native dependencies are locked in `sage-linux-64.lock`. The `sage` wrapper
  activates that environment without changing ordinary `python` or `python3`.
- **Lean/Mathlib 4.34.0** in `/opt/lean` and `/opt/mathlib`. The Dockerfile pins the
  Lean archive checksum and Mathlib commit. Mathlib's manifest pins its transitive
  source dependencies. Compiled caches are fetched during the build, then
  `lake build Mathlib` fills any missing entries before the image is exported.

OpenCode stays at 1.17.11 so changing scientific tools does not also change the
agent harness. Base images are digest-pinned and Debian uses a dated snapshot.
Input locks, the Mathlib manifest/commit and installed package inventories are
preserved in `/opt/image-provenance/`. [Tool usage](tools.md) is also installed at
`/opt/science/README.md` inside each sandbox.

Astropy's IERS data is bundled and automatic downloading is disabled. Accuracy
checks are retained. External databases, pretrained weights and pseudopotentials
are not generally included. No credentials, benchmark datasets or reference
answers are included. OpenSandbox injects execd; network policy is configured by
the sandbox caller, not by the image.

The package list covers scientific computation, CPU ML, image/PDF
and spreadsheet support; unrelated web-service clients and dataset downloaders
are not part of this image. TensorFlow is not installed; CPU JAX and PyTorch are.
Python-MIP is also excluded: its bundled CBC library conflicts with OR-Tools in
the same Python process. Use OR-Tools or CVXPY/HiGHS for integer optimization.
OR-Tools and `highspy` are pinned together because their wheels share a native
HiGHS library name. Updating them requires solving a problem with both import
orders, not just checking the packages separately.

## Refreshing dependencies

Resolve updates deliberately, commit the resulting locks, then rebuild and test.
Do not resolve versions dynamically on sandbox startup. The commands below need
Docker and, for the Sage lock, `jq`:

```sh
docker build --platform linux/amd64 --target base -t opencode-science-builder .
docker run --rm --user "$(id -u):$(id -g)" -e UV_CACHE_DIR=/tmp/uv \
  -v "$PWD:/work" -w /work \
  opencode-science-builder uv --no-config pip compile requirements.in --python-version 3.13.14 \
  --python-platform x86_64-unknown-linux-gnu --torch-backend cpu \
  --generate-hashes --no-header --output-file requirements.lock --upgrade

docker run --rm --platform linux/amd64 -e CONDA_OVERRIDE_ARCHSPEC=x86_64 \
  -e CONDA_OVERRIDE_GLIBC=2.36 \
  -v "$PWD:/inputs:ro" \
  mambaorg/micromamba:2.9.0@sha256:5681ae3caa12844c41d31e1d70f636fbaba31b01bdfd1b5b521742d7e490614d \
  micromamba create --dry-run --json -p /opt/sage \
  -f /inputs/sage-environment.yaml > /tmp/sage-plan.json
jq -er 'if .success then "@EXPLICIT", (.actions.LINK[] | .url + "#" + .sha256) else error("Sage solve failed") end' \
  /tmp/sage-plan.json > sage-linux-64.lock
```

When upgrading Lean, select a Mathlib release first and use the version from its
`lean-toolchain` file. Update both the source commit and archive checksum. Keep
`.lake` compiled files in the final image; deleting them defeats offline use.
`--no-config` keeps dependency resolution independent of Gym's repository-level
uv exclusions and overrides.
The Sage environment pins baseline x86-64 and targets Bookworm's glibc 2.36 rather
than inheriting the resolver container's CPU/libc capabilities. Preserve these
targets when resolving for heterogeneous sandbox workers.

## Validation and publishing

Run representative computations with Docker `--network=none`, including Sage
algebra and a new Lean proof importing Mathlib. Image validation scripts and
registry-specific receipts belong to the deployment workflow, outside Gym.

Push to the authorized registry, then test a fresh sandbox pull and offline
execution before adopting the immutable image digest in benchmark recipes.
Measure cold and repeated tool execution in that sandbox. Kubernetes may load
image files lazily, so local timings do not predict cold sandbox latency. No
package warm-up or runtime installation runs at startup.

BLAS/OpenMP thread counts default to one to avoid oversubscribing small sandboxes;
callers can override the environment for larger CPU allocations.
