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
    # fail_under and precision come from pyproject.toml, just like the serial gate.
    subprocess.run([sys.executable, "-m", "coverage", "report", "--show-missing"], check=True)


if __name__ == "__main__":
    combine(Path(sys.argv[1]))
