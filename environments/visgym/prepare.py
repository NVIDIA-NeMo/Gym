#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare reproducible VisGym datasets without committing generated artifacts.

This is the canonical data-preparation entry point for the VisGym environment.
It delegates to the focused generators kept with the resources server so their
standalone CLIs remain available.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


GYM_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_DIR = GYM_ROOT / "resources_servers" / "visgym" / "scripts"
GENERATORS = {
    "maze": GENERATOR_DIR / "create_maze_curriculum.py",
    "multienv": GENERATOR_DIR / "create_fourteen_env_data.py",
    "hf": GENERATOR_DIR / "create_hf_rl_manifests.py",
}


def _print_help() -> None:
    print(
        """Prepare a VisGym dataset.

Usage:
  python environments/visgym/prepare.py <dataset> [generator options]

Datasets:
  maze      Generate a deterministic online maze curriculum.
  multienv  Generate deterministic multi-environment manifests and assets.
  hf        Build lightweight online-RL manifests from the public VisGym HF data.

Examples:
  python environments/visgym/prepare.py maze --samples-per-stage 1280
  python environments/visgym/prepare.py multienv --combine-envs maze_2d_7x7,maze_3d
  python environments/visgym/prepare.py hf --output-dir /tmp/visgym-hf

Pass --help after the dataset name to see that generator's complete options.
"""
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    dataset = args.pop(0)
    generator = GENERATORS.get(dataset)
    if generator is None:
        choices = ", ".join(GENERATORS)
        raise SystemExit(f"unknown dataset {dataset!r}; choose one of: {choices}")

    return subprocess.run([sys.executable, str(generator), *args], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
