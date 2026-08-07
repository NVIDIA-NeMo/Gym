# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path


SSA_REPO = "https://github.com/strands-labs/benchmark-harnesses.git"
SSA_REVISION = "fd9395b672b670ddb6b90de19723327f007b0655"
_INSTALL_LOCK = threading.Lock()


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


def ensure_ssa(source_root: str | None = None, python: str | None = None) -> Path:
    if python:
        path = Path(python).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"SSA Python does not exist: {path}")
        return path

    with _INSTALL_LOCK:
        cache_root = Path(os.environ.get("NEMO_GYM_SSA_CACHE", "~/.cache/nemo-gym/simple-strands-agent")).expanduser()
        source = Path(source_root) if source_root else cache_root / SSA_REVISION / "source"
        if not source_root and not source.is_dir():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", SSA_REPO, str(source)], check=True)
            subprocess.run(["git", "checkout", SSA_REVISION], cwd=source, check=True)

        package_dir = _package_dir(source)
        source_key = SSA_REVISION
        if source_root:
            digest = hashlib.sha256(str(package_dir).encode()).hexdigest()[:12]
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
        subprocess.run([uv, "venv", "--python", sys.executable, str(venv)], check=True)
        subprocess.run(
            [uv, "pip", "install", "--python", str(python_path), "-e", str(package_dir), "cryptography"],
            check=True,
        )
        subprocess.run(
            [str(python_path), "-c", "from ssa.agent import StrandsResolverAgent"],
            check=True,
        )
        marker.touch()
        return python_path
