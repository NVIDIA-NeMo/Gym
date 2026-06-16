# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-install nvcc into a local prefix at server startup.

The nemo-rl / nemo-gym container ships CUDA runtime libs but **not** the
CUDA Toolkit (no nvcc on PATH). compute-eval requires nvcc to compile and
run the model's solution against hidden tests.

PyPI doesn't have a usable nvcc wheel (``nvidia-cuda-nvcc-cu12`` only ships
``ptxas``). The cleanest no-root install is conda-forge / NVIDIA-channel's
``cuda-nvcc`` package via ``micromamba``. We download a small (~30 MB)
``micromamba`` binary, use it to install ``cuda-nvcc``, ``cuda-cudart-dev``,
and ``cuda-cccl`` into a per-server local prefix, then prepend the install
to ``PATH``, ``LD_LIBRARY_PATH``, and ``CPATH``.

Install paths live in ``.cuda_nvcc/`` next to ``app.py`` so they're tied to
the resources-server venv (cached on lustre via the per-recipe Gym worktree
mount). First boot installs (~5 min); subsequent boots short-circuit on the
existing prefix. To share an install across multiple consumers (e.g. a
NeMo Skills eval client running alongside the Gym resource server), set
``$COMPUTE_EVAL_CUDA_NVCC_PREFIX`` on every consumer to the same prefix —
the first one to boot installs, the rest reuse.

We deliberately do NOT fall back to ``which("nvcc")``: a container shipping
``/usr/local/cuda/bin/nvcc`` at a different CUDA version would silently
win the PATH lookup, and the server would compile against a different
toolchain than any other consumer that explicitly points at the pinned
prefix. Same nvcc on both sides is a parity invariant.

Follows the External Tool Auto-Install pattern from Gym's CLAUDE.md.
"""

from __future__ import annotations

import os
import platform
import subprocess
import urllib.request
from pathlib import Path


# Two supported installation prefixes:
#   * Default: ``.cuda_nvcc/`` next to this file. The server installs into
#     here on first boot and reuses on subsequent boots. When the server's
#     directory lives on shared storage (lustre via a Gym worktree mount),
#     this naturally persists across SLURM jobs.
#   * Override: set $COMPUTE_EVAL_CUDA_NVCC_PREFIX to point at a different
#     prefix. This is how multiple servers / pipelines (e.g. a NeMo Skills
#     eval client running alongside the Gym resource server) share a single
#     install — every consumer points at the same prefix and gets bit-
#     identical nvcc.
_LOCAL_PREFIX_DEFAULT = Path(__file__).parent / ".cuda_nvcc"


def _resolve_default_prefix() -> Path:
    env_override = os.environ.get("COMPUTE_EVAL_CUDA_NVCC_PREFIX")
    if env_override:
        return Path(env_override)
    return _LOCAL_PREFIX_DEFAULT


DEFAULT_PREFIX = _resolve_default_prefix()

# Pin a known-good cuda-nvcc version — must be compatible with the host
# driver (CUDA 12.x). aws-dfw + the Nemotron-3-Nano-30B vLLM container use
# CUDA 12.x at runtime, so 12.9 is safe.
CUDA_PKG_SPEC = (
    "cuda-nvcc=12.9",
    "cuda-cudart-dev=12.9",
    "cuda-cccl=12.9",
    # Hidden-test code in nvidia/compute-eval problems calls nvtxRangePushA;
    # without cuda-nvtx-dev the test_main.cu binaries fail to compile/link
    # regardless of what the model generates.
    "cuda-nvtx-dev=12.9",
)
CONDA_CHANNELS = ("nvidia/label/cuda-12.9.0", "conda-forge")

# Micromamba binary download URL (architecture-dependent).
_MAMBA_URL_TMPL = "https://micro.mamba.pm/api/micromamba/{platform}/latest"


def _detect_mamba_platform() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "linux-64"
    if machine in ("aarch64", "arm64"):
        return "linux-aarch64"
    raise RuntimeError(f"unsupported platform for cuda-nvcc auto-install: {machine}")


def _download_micromamba(target: Path) -> Path:
    mamba_path = target / "bin" / "micromamba"
    if mamba_path.exists():
        return mamba_path
    target.mkdir(parents=True, exist_ok=True)
    url = _MAMBA_URL_TMPL.format(platform=_detect_mamba_platform())
    print(f"[setup_cuda_nvcc] downloading micromamba from {url}")
    archive = target / "micromamba.tar.bz2"
    urllib.request.urlretrieve(url, archive)
    # The micromamba archive is a tarball with bin/micromamba inside.
    subprocess.run(
        ["tar", "-xjf", str(archive), "-C", str(target), "bin/micromamba"],
        check=True,
    )
    archive.unlink()
    mamba_path.chmod(0o755)
    return mamba_path


def _run_micromamba_install(mamba: Path, prefix: Path) -> None:
    env_dir = prefix / "env"
    # `micromamba install` requires the env to exist; `create` makes it
    # first. We always go through `create` here because if env_dir already
    # existed with a valid nvcc, ensure_cuda_nvcc() short-circuited
    # upstream.
    print(f"[setup_cuda_nvcc] creating env with {CUDA_PKG_SPEC} at {env_dir}")
    cmd = [
        str(mamba),
        "create",
        "--yes",
        "--root-prefix",
        str(prefix / "mamba_root"),
        "--prefix",
        str(env_dir),
    ]
    for ch in CONDA_CHANNELS:
        cmd.extend(["-c", ch])
    cmd.extend(CUDA_PKG_SPEC)
    subprocess.run(cmd, check=True)


def _wire_paths(env_dir: Path) -> Path:
    nvcc = env_dir / "bin" / "nvcc"
    if not nvcc.exists():
        raise RuntimeError(f"nvcc not found at {nvcc} after micromamba install")

    # conda's cuda packages put libcudart + cuda_runtime.h under
    # targets/<arch>-linux/{lib,include}. Auto-detect the target dir
    # since arch naming varies (x86_64-linux, sbsa-linux for ARM64 SBSA,
    # aarch64-linux on some distros).
    targets_root = env_dir / "targets"
    target_dirs = sorted(p for p in targets_root.iterdir() if p.is_dir()) if targets_root.exists() else []
    if not target_dirs:
        raise RuntimeError(f"no targets/*-linux dirs under {targets_root}")
    target = target_dirs[0]  # take whichever conda created

    bin_dir = str(env_dir / "bin")
    lib_dir = str(env_dir / "lib")
    inc_dir = str(env_dir / "include")
    targets_lib = str(target / "lib")
    targets_inc = str(target / "include")

    def _prepend(var: str, *vals: str) -> None:
        existing = os.environ.get(var, "")
        new = ":".join(list(vals) + ([existing] if existing else []))
        os.environ[var] = new

    _prepend("PATH", bin_dir)
    _prepend("LD_LIBRARY_PATH", targets_lib, lib_dir)
    _prepend("CPATH", targets_inc, inc_dir)
    _prepend("LIBRARY_PATH", targets_lib, lib_dir)  # for nvcc linker search at compile time
    os.environ["CUDA_HOME"] = str(env_dir)  # absolute, not list-prepend

    return nvcc


def ensure_cuda_nvcc(prefix: Path = DEFAULT_PREFIX) -> Path:
    """Return a usable nvcc path. Installs into ``prefix`` if necessary.

    Only two supported paths:

      1. ``prefix/env/bin/nvcc`` exists → reuse (wires PATH/LD_LIBRARY_PATH).
      2. Otherwise → install via micromamba into ``prefix``.

    No fallback to ``PATH``-nvcc. A container that happens to ship
    ``/usr/local/cuda/bin/nvcc`` at a different CUDA version would otherwise
    silently win the lookup, and the server would compile against a
    different toolchain than any other consumer that points at the same
    pinned prefix (e.g. a NeMo Skills eval client sourcing the same env).
    Compiling against the SAME nvcc on both sides is a parity invariant;
    we'd rather pay a one-time install than risk silent version drift.
    """
    env_dir = prefix / "env"
    nvcc = env_dir / "bin" / "nvcc"
    if nvcc.exists():
        print(f"[setup_cuda_nvcc] reusing existing install at {env_dir}")
        return _wire_paths(env_dir)

    mamba = _download_micromamba(prefix)
    _run_micromamba_install(mamba, prefix)
    return _wire_paths(env_dir)


if __name__ == "__main__":
    nvcc = ensure_cuda_nvcc()
    print(f"nvcc: {nvcc}")
    subprocess.run([str(nvcc), "--version"], check=True)
