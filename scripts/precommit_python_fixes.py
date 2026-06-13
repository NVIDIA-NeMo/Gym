#!/usr/bin/env python3
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
"""Local pre-commit text fixers for Python files.

These replace the small subset of pre-commit/pre-commit-hooks used by this
repository, avoiding a git clone of that hook repository during CI setup.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def fix_end_of_file(data: bytes) -> bytes:
    if not data:
        return data
    newline = b"\r\n" if data.endswith(b"\r\n") else b"\n"
    return data.rstrip(b"\r\n") + newline


def fix_trailing_whitespace(data: bytes) -> bytes:
    return re.sub(rb"[ \t]+(?=\r?\n|\Z)", b"", data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--end-of-file", action="store_true")
    mode.add_argument("--trailing-whitespace", action="store_true")
    parser.add_argument("files", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed = []
    fixer = fix_end_of_file if args.end_of_file else fix_trailing_whitespace

    for filename in args.files:
        path = Path(filename)
        original = path.read_bytes()
        fixed = fixer(original)
        if fixed != original:
            path.write_bytes(fixed)
            changed.append(filename)

    for filename in changed:
        print(f"Fixed {filename}")

    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
