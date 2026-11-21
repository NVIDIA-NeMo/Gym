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
"""Trusted, isolated-interpreter preflight in the fresh SWE-Gym verifier image.

Run before any candidate code or repository install. The input is generated on the
resources server, never copied from the agent workspace. No expected outputs belong here.
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path


EXIT_UNSAFE_PATH = 95
EXIT_WRONG_BASE = 94


def check_paths(repo: Path, paths: list[str]) -> None:
    """Reject existing symlinks and special files along every validated candidate path."""
    if repo.is_symlink() or not repo.is_dir():
        raise ValueError("repository root must be a real directory")
    for path in paths:
        parts = path.split("/")
        if not path or any(part in {"", ".", ".."} for part in parts) or path.startswith("/"):
            raise ValueError("noncanonical candidate path")
        current = repo
        for index, part in enumerate(parts):
            current = current / part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                break
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) if index < len(parts) - 1 else stat.S_ISREG(mode)):
                raise ValueError(f"candidate path crosses a symlink or special file: {path}")


def main(path: Path) -> int:
    request = json.loads(path.read_text(encoding="utf-8"))
    repo = Path(request["repo"])
    # The fresh image, not the agent's refs or working directory, is the source of authority.
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=True
    ).stdout.strip()
    if head != request["base_commit"]:
        print("fresh verifier image is not at the trusted task base", file=sys.stderr)
        return EXIT_WRONG_BASE
    try:
        check_paths(repo, request["paths"])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_UNSAFE_PATH
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
