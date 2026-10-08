# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream runtime, kept separate from Gym's Python 3.13 environment."""

import asyncio
import os
import signal
import subprocess
from pathlib import Path


UPSTREAM_URL = "https://github.com/BenchCAD/BenchCAD-main.git"
UPSTREAM_REVISION = "77fd16a80a4e5fc39964ee23b6cc3225beefac8f"
TASKS = ("vision2code", "codeedit", "vision_qa", "code_qa")
WORKER = Path(__file__).with_name("worker.py")
CAD_REQUIREMENTS = Path(__file__).with_name("cad-requirements.txt")
CAD_CONSTRAINTS = Path(__file__).with_name("cad-constraints.txt")


def ensure_runtime(root: Path) -> Path:
    """Fetch upstream and install its CAD pins with the upstream ARM nlopt fix."""
    root = root.resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", UPSTREAM_URL, str(root)], check=True)
        subprocess.run(["git", "checkout", UPSTREAM_REVISION], cwd=root, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode(errors="replace").strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"BenchCAD checkout must be at {UPSTREAM_REVISION}; found {revision} at {root}")
    python = root / ".venv/bin/python"
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(root / ".venv")], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(CAD_REQUIREMENTS),
            "--override",
            str(CAD_CONSTRAINTS),
        ],
        check=True,
    )
    return python


async def run_worker(python: Path, root: Path, *args: str, timeout: float = 900) -> str:
    """Run trusted preparation/scoring; reap descendants on cancellation or timeout."""
    env = os.environ.copy()
    # Camera ablations must never silently alter the canonical evaluation input.
    env.pop("BENCH_VIEW_HINT", None)
    process = await asyncio.create_subprocess_exec(
        str(python),
        str(WORKER),
        "--root",
        str(root),
        *args,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"BenchCAD worker exited {process.returncode}: {stderr.decode(errors='replace')[-4000:]}")
    return stdout.decode(errors="replace")
