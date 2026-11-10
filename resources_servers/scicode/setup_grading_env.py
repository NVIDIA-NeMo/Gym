# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build SciCode's pinned scientific stack in a separate interpreter."""

import fcntl
import hashlib
import shutil
import subprocess
from pathlib import Path


REQUIREMENTS = Path(__file__).with_name("grading-requirements.lock")
PYTHON_VERSION = "3.11"


def ensure_grading_env(venv_path: Path) -> Path:
    """Create or update the isolated grader; serialize concurrent server starts."""
    venv_path = venv_path.absolute()
    python = venv_path / "bin" / "python"
    marker = venv_path / ".requirements-sha256"
    fingerprint = hashlib.sha256(REQUIREMENTS.read_bytes() + PYTHON_VERSION.encode()).hexdigest()
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    with venv_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if python.is_file() and marker.is_file() and marker.read_text() == fingerprint:
            return python
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("SciCode grading requires uv on PATH or explicit grading_interpreters.")
        marker.unlink(missing_ok=True)
        # Python-version changes and removed requirements require a clean environment.
        subprocess.run([uv, "venv", "--clear", "--python", PYTHON_VERSION, str(venv_path)], check=True)
        subprocess.run([uv, "pip", "sync", "--python", str(python), str(REQUIREMENTS)], check=True)
        # Check APIs the original benchmark uses, including ones missing in modern SciPy.
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "from scipy.integrate import simps; from scipy.linalg import eigh; "
                "import numpy, h5py, matplotlib, sympy, networkx; "
                "assert abs(simps([0., 1., 4.], [0., 1., 2.]) - 8 / 3) < 1e-12",
            ],
            check=True,
        )
        marker.write_text(fingerprint)
    return python
