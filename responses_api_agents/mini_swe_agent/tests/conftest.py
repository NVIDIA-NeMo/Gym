# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the sandbox runtime's packages on this host, so the tests of real mini-SWE 2.4.6 and the MCP 2 CLI run.

When ``uv`` is on PATH and ``MINI_SWE_RUNTIME_PYTHON`` is unset, a venv is installed from the runtime lock before
collection and exported in that variable, which the runtime tests read at import. The venv is keyed by the lock, so a
relock builds a new one, and it is published by a rename, so concurrent test sessions never use a partial one.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


RUNTIME_FILES = Path(__file__).parents[1] / "runtime"
RUNTIME_INPUTS = ("requirements.in", "requirements.txt", "overrides.txt")


def pytest_configure(config: pytest.Config) -> None:
    uv = shutil.which("uv")
    if os.environ.get("MINI_SWE_RUNTIME_PYTHON"):
        return
    if uv is None:
        # The runtime tests would otherwise skip without saying the real mini-SWE runtime was never exercised.
        raise pytest.UsageError("mini-SWE tests need uv on PATH, or MINI_SWE_RUNTIME_PYTHON set to a 2.4.6 runtime")
    key = hashlib.sha256(b"".join((RUNTIME_FILES / name).read_bytes() for name in RUNTIME_INPUTS)).hexdigest()[:16]
    target = Path(tempfile.gettempdir()) / f"nemo-gym-mini-swe-test-runtime-{key}"
    if not target.exists():
        building = Path(tempfile.mkdtemp(prefix=f"{target.name}.", dir=target.parent))
        try:
            subprocess.run(
                [uv, "--no-config", "venv", "--quiet", "--allow-existing", "--python", sys.executable, str(building)],
                check=True,
            )
            subprocess.run(
                [
                    uv,
                    "--no-config",
                    "pip",
                    "install",
                    "--quiet",
                    "--python",
                    str(building / "bin" / "python"),
                    "-r",
                    "requirements.in",
                    "-c",
                    "requirements.txt",
                    "--override",
                    "overrides.txt",
                ],
                cwd=RUNTIME_FILES,
                check=True,
            )
            building.rename(target)
        except OSError:
            # Another test session published it first.
            if not target.exists():
                raise
        finally:
            shutil.rmtree(building, ignore_errors=True)
    os.environ["MINI_SWE_RUNTIME_PYTHON"] = str(target / "bin" / "python")
