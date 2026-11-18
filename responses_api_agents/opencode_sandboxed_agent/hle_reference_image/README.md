# HLE-Diamond reference sandbox image

The code sandbox that upstream's with-tools protocol prescribes, with OpenCode added as the harness:
Python 3.11.16, a shell, the twelve pinned packages of upstream's `requirements-tools.txt`, the
OpenCode binary, and nothing else. It replaces the
[offline scientific image](../offline_science_image/README.md) (Python 3.13, SageMath, Lean/Mathlib,
solvers, OCR) for `benchmarks/hle_diamond/opencode_reference.yaml`, whose README states what the
preset still does not reproduce.

**Protocol source.** [`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) at commit
`22ed3074b1e7b134bcbc09028d0ba320839b0655`: <!-- pragma: allowlist secret -->

| Upstream file | SHA-256 of the snapshot | Used as |
| --- | --- | --- |
| `docs/evaluation-with-tools.md` | `3852a7f7008ceb27e9aec97f223436acbecdc9a4bc22cd63de54e5f0d49d9cab` <!-- pragma: allowlist secret --> | section 1 (no direct internet) is the preset's `network_access: model_and_tools`; section 2 (Python 3.11.16, pinned requirements, no specialist solvers, chess engines, OCR or task libraries) is this image |
| `docs/requirements-tools.txt` | `c1c626beb8f2fab11609f6ed41a2721d9acd139554d6ce3c8c83f52a49797068` <!-- pragma: allowlist secret --> | [`requirements-tools.txt`](requirements-tools.txt), copied verbatim; `tests/test_hle_reference_image.py` pins the hash |

Build from this directory (needs network access for the pinned downloads; the sandbox performs none):

```sh
docker build --platform linux/amd64 -t gym-opencode-hle-reference:local .
```

## Contents

| Layer | Pin | Why |
| --- | --- | --- |
| `python:3.11.16-slim-bookworm` | image index digest in the Dockerfile | upstream's interpreter version; Debian packages come from the dated snapshot `20260916T000000Z` |
| `requirements-tools.txt` | upstream's text, verbatim | the twelve direct requirements at upstream's versions |
| `requirements.lock` | resolved from it for CPython 3.11.16 / linux x86_64 with artifact hashes | the 22 transitive dependencies, installed with `--require-hashes` so a rebuild cannot drift |
| OpenCode 1.17.11 | release tarball URL and SHA-256, identical to the science image | the harness |
| `@opencode-ai/plugin` 1.17.11 | `opencode/package-lock.json`, identical to the science image, installed with `npm ci` at build time into `/root/.config/opencode` | OpenCode's plugin dependency, preinstalled so the sandbox never fetches it; Node is a build stage only |
| `ripgrep` | Debian bookworm (13.0.0) | OpenCode's `glob` and `grep` tools call `rg` and otherwise try to download it, which the reference network policy forbids |

Nothing else: no SageMath, Lean/Mathlib, Z3, RDKit, OCR, chess engines, compilers, `git`, `curl` or
Node in the final image. `pip`, `setuptools` and `wheel` are the base image's own and have no network
to use. Provenance is kept under `/opt/image-provenance/` (`requirements-tools.txt`,
`requirements.lock`, `python-installed.txt` from `pip freeze`, `debian-installed.txt`). No
credentials, benchmark rows or reference answers are included; network policy is the sandbox
caller's, not the image's.

## Offline verification

Run on 2026-10-26 against the local build (image id `sha256:84c8bbd2357d96f8c388151ecc3ed4a95e9ae5d422b71adc0f909a1e4b4e1b9e`, <!-- pragma: allowlist secret -->
785 MB, `docker run --rm --network=none`):

- `python --version` printed `Python 3.11.16`; `opencode --version` printed `1.17.11`.
- `pip freeze` listed exactly the 34 distributions of `requirements.lock` at the locked versions: the
  twelve upstream pins (`matplotlib==3.11.2`, `mpmath==1.3.0`, `networkx==3.6.1`, `numpy==2.4.6`,
  `pandas==3.0.6`, `pillow==12.3.0`, `PyYAML==6.0.3`, `requests==2.34.2`, `scikit-learn==1.9.1`,
  `scipy==1.17.1`, `statsmodels==0.15.0`, `sympy==1.14.0`) and their 22 dependencies; `pip`,
  `setuptools` and `wheel` are the only other distributions (`pip list`).
- `which sage lean lake z3 stockfish tesseract gcc node npm curl git` found nothing; importing `z3`,
  `rdkit`, `sage`, `chess`, `pytesseract`, `cv2`, `torch`, `jax`, `pysat` and `cvxpy` fails.
- `which opencode rg bash python3` gave `/usr/local/bin/opencode`, `/usr/bin/rg`, `/usr/bin/bash`,
  `/usr/local/bin/python3`; the twelve packages import.
- `urllib.request.urlopen("https://example.com")` fails (`URLError`) under `--network=none`. This
  proves only the Docker flag: egress enforcement for a run comes from the sandbox provider's
  network policy (see the gate below).

## Refreshing the lock

Resolve deliberately, commit the lock, rebuild and re-run the checks above. `--no-config` keeps the
resolution independent of Gym's repository-level `uv` settings:

```sh
uv --no-config pip compile requirements-tools.txt --python-version 3.11.16 \
  --python-platform x86_64-unknown-linux-gnu --generate-hashes --no-header \
  --output-file requirements.lock
```

When upstream changes `requirements-tools.txt`, copy the new file verbatim, update the SHA-256 in
this README and in `tests/test_hle_reference_image.py`, and re-resolve the lock.

## Open gate: OpenSandbox

The image has run only on the local Docker provider, where egress is not restricted, through the
acknowledged `sandbox_docker_local.yaml` form (see the
[benchmark README](../../../benchmarks/hle_diamond/README.md#reference-preset)). Only the
OpenSandbox provider enforces `network_access: model_and_tools`; no OpenSandbox run of this image
exists yet, so no run of it is protocol-grade. Before publishing: push to the authorised registry,
pull into a fresh OpenSandbox sandbox, confirm a direct fetch is blocked while the tool servers are
reachable, and adopt the immutable digest in `OPENCODE_REFERENCE_SANDBOX_IMAGE`.
