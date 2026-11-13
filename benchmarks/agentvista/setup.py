# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch and verify the public upstream benchmark dependency; no model requests."""

import argparse
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responses_api_agents.agentvista_agent.runtime.source import source_pin, verify_source


def provision_source(destination: Path) -> None:
    pin = source_pin()
    if not destination.exists():
        subprocess.run(["git", "clone", "--no-checkout", pin["repository"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", pin["revision"]], check=True)
    verify_source(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    provision_source(parser.parse_args().source.resolve())
