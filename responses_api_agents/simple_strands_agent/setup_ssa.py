# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from filelock import FileLock


SSA_REPO = "https://github.com/strands-labs/benchmark-harnesses.git"
SSA_REVISION = "fd9395b672b670ddb6b90de19723327f007b0655"  # pragma: allowlist secret


def _package_dir(source_root: Path) -> Path:
    source_root = source_root.expanduser().resolve()
    if (source_root / "pyproject.toml").is_file() and (source_root / "src" / "ssa").is_dir():
        return source_root
    package_dir = source_root / "simple-strands-agent"
    if (package_dir / "pyproject.toml").is_file():
        return package_dir
    raise RuntimeError(f"Simple Strands Agent package not found under {source_root}")


def _python_path(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _workspace_dir(package_dir: Path) -> Path:
    for candidate in (package_dir, package_dir.parent):
        if (candidate / "uv.lock").is_file():
            return candidate
    raise RuntimeError(f"Simple Strands Agent lockfile not found for {package_dir}")


def _revision(source: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _prepare_source(source: Path) -> None:
    if source.is_dir() and _revision(source) == SSA_REVISION:
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=source.parent) as temp_dir:
        clone = Path(temp_dir) / "source"
        subprocess.run(["git", "clone", SSA_REPO, str(clone)], check=True)
        subprocess.run(["git", "checkout", SSA_REVISION], cwd=clone, check=True)
        if source.exists():
            shutil.rmtree(source)
        clone.rename(source)


def ensure_ssa(source_root: str | None = None, python: str | None = None) -> Path:
    if python:
        path = Path(python).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"SSA Python does not exist: {path}")
        return path

    cache_root = Path(os.environ.get("NEMO_GYM_SSA_CACHE", "~/.cache/nemo-gym/simple-strands-agent")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    with FileLock(cache_root / ".install.lock"):
        source = Path(source_root) if source_root else cache_root / SSA_REVISION / "source"
        if not source_root:
            _prepare_source(source)

        package_dir = _package_dir(source)
        workspace_dir = _workspace_dir(package_dir)
        lock_digest = hashlib.sha256((workspace_dir / "uv.lock").read_bytes()).hexdigest()[:12]
        source_key = f"{SSA_REVISION}-{lock_digest}"
        if source_root:
            digest = hashlib.sha256(str(package_dir).encode() + lock_digest.encode()).hexdigest()[:12]
            source_key = f"local-{digest}"
        venv = cache_root / source_key / f"venv-{sys.version_info.major}.{sys.version_info.minor}"
        python_path = _python_path(venv)
        marker = venv / ".complete"
        if marker.is_file() and python_path.is_file():
            return python_path

        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to install Simple Strands Agent")
        venv.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ | {"UV_PROJECT_ENVIRONMENT": str(venv)}
        env.pop("VIRTUAL_ENV", None)
        subprocess.run(
            [
                uv,
                "sync",
                "--project",
                str(workspace_dir),
                "--package",
                "simple-strands-agent",
                "--locked",
                "--no-dev",
                "--python",
                sys.executable,
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [str(python_path), "-c", "from ssa.agent import StrandsResolverAgent"],
            check=True,
        )
        marker.touch()
        return python_path
