#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require all four core inputs and sandbox coverage before enforcing the configured gate."""

import subprocess
import sys
from pathlib import Path


def combine(directory: Path) -> None:
    inputs = [directory / f".coverage.core-{index}" for index in range(4)]
    inputs.append(directory / ".coverage.sandbox")
    for path in inputs:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or empty coverage input: {path}")
    subprocess.run([sys.executable, "-m", "coverage", "combine", "--keep", *map(str, inputs)], check=True)
    from coverage import Coverage

    coverage = Coverage()
    coverage.load()
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0"))
    # Shards can execute dependency-cache helpers and synthetic code (e.g. x.py).
    # Those sources aren't portable artifacts. Keep every measured repository
    # source, including tracked files that are missing, so reporting fails closed.
    sources = sorted(coverage.get_data().measured_files() & tracked)
    if not sources:
        raise ValueError("No tracked repository sources in combined coverage")
    # fail_under, precision and exclusions remain owned by pyproject.toml.
    subprocess.run([sys.executable, "-m", "coverage", "report", "--show-missing", *sources], check=True)


if __name__ == "__main__":
    combine(Path(sys.argv[1]))
