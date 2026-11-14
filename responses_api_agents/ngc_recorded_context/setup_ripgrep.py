# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adopt the component's locked ripgrep installation before policy inference."""

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path


def ensure_ripgrep() -> str:
    """Reuse PATH or expose the executable installed by the component dependencies."""
    executable = shutil.which("rg")
    if executable is None:
        binary = Path(sysconfig.get_path("scripts")) / ("rg.exe" if os.name == "nt" else "rg")
        if not binary.is_file():
            raise RuntimeError(
                "The replay requires ripgrep before inference. Install its locked component dependencies with "
                "`uv sync --locked --project responses_api_agents/ngc_recorded_context`."
            )
        os.environ["PATH"] = str(binary.parent) + os.pathsep + os.environ.get("PATH", "")
        executable = str(binary)
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10, check=True)
        if not result.stdout.startswith("ripgrep "):
            raise ValueError("Unexpected rg executable")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RuntimeError(
            "The replay's ripgrep installation is unusable; reinstall its locked dependencies"
        ) from error
    return executable
