# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision the original pinned benchmark dependency, not a second integration checkout."""

import argparse
import os
import subprocess
from pathlib import Path

from responses_api_agents.wildclawbench_agent.data import source_pin, verify_source


def provision_source(destination: Path) -> None:
    pin = source_pin()
    if not destination.exists():
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(["git", "clone", "--no-checkout", pin["repository"], str(destination)], env=env, check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", pin["revision"]], env=env, check=True)
    verify_source(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    provision_source(args.source.absolute())
