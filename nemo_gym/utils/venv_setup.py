# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serialize component setup without importing Gym or its dependencies."""

import argparse
import fcntl
import subprocess
from pathlib import Path


SETUP_COMPLETE_MARKER = ".nemo-gym-setup-complete"


def setup_environment(venv: Path, command: str, *, skip_if_ready: bool) -> int:
    venv = venv.resolve()
    venv.parent.mkdir(parents=True, exist_ok=True)
    marker = venv / SETUP_COMPLETE_MARKER
    required_files = [venv / "bin/python", venv / "bin/activate"]
    # Keep the lock outside the venv, and never unlink it: uv may recreate the
    # environment, and another installer may already be waiting on this inode.
    with venv.with_name(f"{venv.name}.setup.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for virtual environment setup: {venv}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)

        if skip_if_ready and marker.is_file() and all(path.is_file() for path in required_files):
            return 0

        marker.unlink(missing_ok=True)
        # The installer inherits the lock, so killing this wrapper cannot let
        # another setup overlap a surviving installer. Close, rather than
        # explicitly unlock, our descriptor when leaving this block.
        result = subprocess.run(["/bin/bash", "-c", command], pass_fds=(lock.fileno(),), check=False)
        if result.returncode:
            return result.returncode if result.returncode > 0 else 128 - result.returncode
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise RuntimeError(f"Virtual environment setup did not create: {', '.join(missing)}")
        marker.touch()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--skip-if-ready", action="store_true")
    args = parser.parse_args()
    return setup_environment(args.venv, args.command, skip_if_ready=args.skip_if_ready)


if __name__ == "__main__":
    raise SystemExit(main())
