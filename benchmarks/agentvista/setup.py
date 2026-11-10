# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch and verify the pinned original benchmark dependency; no model requests."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def verify_source(source: Path) -> None:
    """Check every original execution, prompt, tool and judge file against its pin."""
    pin = json.loads(Path(__file__).with_name("source-pin.json").read_text())
    for name, expected in pin["files"].items():
        path = source / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"AgentVista source differs from pinned revision: {name}")


def provision_source(destination: Path) -> None:
    """Clone a fresh pinned source or verify an existing source without modifying it."""
    pin = json.loads(Path(__file__).with_name("source-pin.json").read_text())
    if not destination.exists():
        subprocess.run(["git", "clone", "--no-checkout", pin["repository"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", pin["revision"]], check=True)
    verify_source(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    options = parser.parse_args()
    provision_source(options.source.resolve())
