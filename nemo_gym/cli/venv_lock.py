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

"""Run server virtual-environment setup under a cross-process file lock."""

import argparse
import fcntl
import subprocess
from pathlib import Path


def run_locked(lock_path: Path, command: str) -> int:
    """Run ``command`` after exclusively locking ``lock_path``."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for another process to finish setting up this virtual environment: {lock_path}")
            fcntl.flock(lock_file, fcntl.LOCK_EX)

        # Keep the descriptor inherited by the setup shell. If this wrapper is
        # terminated, an in-flight installer continues to hold the lock until
        # it exits instead of racing a newly started setup.
        return subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            check=False,
            pass_fds=(lock_file.fileno(),),
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock_path", type=Path)
    parser.add_argument("command")
    args = parser.parse_args()
    return run_locked(args.lock_path, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
