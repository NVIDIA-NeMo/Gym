# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolate MathArena's ANTLR 4.11 from Gym/OmegaConf's incompatible ANTLR 4.9."""

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
UPSTREAM_REVISION = "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b"


def ensure_parser_runtime(venv_path: Path) -> Path:
    """Create a minimal pinned parser-only venv once, safely across concurrent servers."""
    venv_path = venv_path.resolve()
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    requirements = DIRECTORY / "parser-requirements.txt"
    digest = hashlib.sha256()
    for path in [
        requirements,
        DIRECTORY / "parser.py",
        DIRECTORY / "worker.py",
        *sorted((DIRECTORY / "_vendor").glob("*.txt")),
    ]:
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    python = venv_path / "bin" / "python"
    marker = venv_path / ".installed"
    with (venv_path.parent / f"{venv_path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if python.is_file() and marker.is_file() and marker.read_text() == fingerprint:
            return python
        uv = shutil.which("uv")
        if uv:
            if not python.is_file():
                subprocess.run([uv, "venv", "--python", sys.executable, str(venv_path)], check=True)
            install = [uv, "pip", "install", "--python", str(python), "--no-deps", "-r", str(requirements)]
        else:
            if not python.is_file():
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            # A previous uv-created environment may have no pip. A later launch
            # without uv must still be able to refresh a changed source snapshot.
            subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
            install = [str(python), "-m", "pip", "install", "--no-deps", "-r", str(requirements)]
        subprocess.run(install, check=True)
        checked = subprocess.run(
            [str(python), str(DIRECTORY / "worker.py")],
            input=json.dumps({"text": r"\boxed{42}", "strict": False, "expected_answer": 42}),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        result = json.loads(checked.stdout)
        if not result.get("valid") or result.get("reward") != 1.0:
            raise RuntimeError(f"MathArena parser self-test failed: {result}")
        marker.write_text(fingerprint)
    return python
