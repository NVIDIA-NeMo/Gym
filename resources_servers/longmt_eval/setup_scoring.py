# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision LongMT's locked Python 3.12 evaluator independently of Gym."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import subprocess
import time
from pathlib import Path


LOG = logging.getLogger(__name__)
PROJECT = Path(__file__).parent / "scoring"


def ensure_scoring_python(cache: Path) -> Path:
    """Install once per lockfile, or validate an explicitly prebuilt evaluator."""
    override = os.environ.get("LONGMT_SCORING_PYTHON")
    if override:
        python = Path(override).absolute()
        validate_python(python)
        return python
    fingerprint = hashlib.sha256(b"managed-python-v1\0" + (PROJECT / "uv.lock").read_bytes()).hexdigest()
    root = cache / fingerprint[:16]
    root.mkdir(parents=True, exist_ok=True)
    python = root / ".venv" / "bin" / "python"
    with (root / "setup.lock").open("w") as handle:
        LOG.info("Waiting for LongMT scoring setup lock: %s", root)
        deadline = time.monotonic() + 1800
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"LongMT scoring setup lock timed out: {root}")
                LOG.info("LongMT scoring setup still waiting for lock: %s", root)
                time.sleep(10)
        marker = root / "ready"
        if not marker.exists():
            LOG.info("Installing locked Python 3.12 scoring environment at %s", root)
            subprocess.run(
                [
                    "uv",
                    "sync",
                    "--locked",
                    "--no-dev",
                    "--managed-python",
                    "--python",
                    "3.12",
                    "--project",
                    str(PROJECT),
                ],
                env={
                    **os.environ,
                    "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
                    "UV_PYTHON_INSTALL_DIR": str(root / "python"),
                },
                timeout=1800,
                check=True,
            )
            validate_python(python)
            marker.write_text(fingerprint + "\n")
        elif not python.exists():
            raise RuntimeError(f"Scoring environment is incomplete: {root}; use a new cache directory")
    return python


def validate_python(python: Path) -> None:
    """Fail before allocating actors if Python or evaluator imports are incompatible."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import sys; assert sys.version_info[:2] == (3, 12), 'LongMT scoring requires Python 3.12'; "
            "import torch, segale_align, comet, laser_encoders, ersatz.split",
        ],
        env=env,
        check=True,
        timeout=120,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(ensure_scoring_python(args.cache))
