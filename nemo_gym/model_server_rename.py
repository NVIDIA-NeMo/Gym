# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Rewrite references to a renamed model server across configs and docs.

Renaming a model server (RFC M6b / friction #9, e.g. ``vllm_model``→``vllm_endpoint`` or
``local_vllm_model``→``vllm_server``) touches references in many files. This tool rewrites the two
*textual* reference forms that are safe to transform mechanically:

1. ``config_paths`` / dotted-override **directory paths**: ``responses_api_models/<old>/...``
   → ``responses_api_models/<new>/...``.
2. The server's **config key** under ``responses_api_models:`` — a line ``<indent><old>:`` whose
   parent is ``responses_api_models:`` — renamed to ``<new>:``.

It deliberately does NOT touch Python imports (``from responses_api_models.<old>.app import ...``)
or move directories — those are done as part of the dir-move/compat-shim step, not by a text pass.
Run via ``python -m nemo_gym.model_server_rename <old> <new> <paths...> [--dry-run]``.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


_PARENT_KEY = "responses_api_models:"


def rename_references(text: str, old: str, new: str) -> Tuple[str, int]:
    """Rewrite directory-path and config-key references from ``old`` to ``new`` in ``text``.

    Returns the rewritten text and the number of substitutions made. Path references
    (``responses_api_models/<old>``) are rewritten anywhere; the bare ``<old>:`` key is rewritten
    only when it is indented directly under a ``responses_api_models:`` line, so unrelated keys
    that merely share the name are left alone.
    """
    count = 0
    path_pattern = re.compile(rf"(responses_api_models/){re.escape(old)}(?=[/\"'\],\s]|$)")

    out_lines: List[str] = []
    parent_indent: int = -1  # indent of the most recent `responses_api_models:` line, else -1
    key_pattern = re.compile(rf"^(?P<indent>\s*){re.escape(old)}:\s*$")
    # `_delete_key: <old>` removes an inherited server block by name (the key == the dir name), so it
    # tracks the rename. Only rewritten when scoped directly under a `responses_api_models:` block.
    delete_key_pattern = re.compile(rf"^(?P<indent>\s*)_delete_key:\s*{re.escape(old)}\s*$")

    for line in text.split("\n"):
        new_line, n = path_pattern.subn(rf"\1{new}", line)
        count += n

        stripped = new_line.strip()
        indent = len(new_line) - len(new_line.lstrip())
        key_match = key_pattern.match(new_line)
        delete_match = delete_key_pattern.match(new_line)
        if key_match is not None and parent_indent >= 0 and indent == parent_indent + 2:
            new_line = f"{key_match.group('indent')}{new}:"
            count += 1
        elif delete_match is not None and parent_indent >= 0 and indent == parent_indent + 2:
            new_line = f"{delete_match.group('indent')}_delete_key: {new}"
            count += 1

        # Track whether we're directly under a `responses_api_models:` block.
        if stripped == _PARENT_KEY:
            parent_indent = indent
        elif stripped and indent <= parent_indent:
            parent_indent = -1

        out_lines.append(new_line)

    return "\n".join(out_lines), count


def rename_file(path: Path, old: str, new: str, dry_run: bool = False) -> int:
    """Rewrite a single file in place. Returns the number of substitutions."""
    original = path.read_text()
    rewritten, count = rename_references(original, old, new)
    if count and not dry_run:
        path.write_text(rewritten)
    return count


def _iter_files(paths: List[Path]):  # pragma: no cover
    for path in paths:
        if path.is_dir():
            for pattern in ("*.yaml", "*.yml", "*.md", "*.sh"):
                yield from sorted(path.rglob(pattern))
        else:
            yield path


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="Current model-server name, e.g. local_vllm_model")
    parser.add_argument("new", help="New model-server name, e.g. vllm_server")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to rewrite.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()

    total_files = 0
    total = 0
    for path in _iter_files(args.paths):
        n = rename_file(path, args.old, args.new, dry_run=args.dry_run)
        if n:
            total_files += 1
            total += n
            verb = "would rewrite" if args.dry_run else "rewrote"
            print(f"{verb} {n} ref(s): {path}")

    verb = "Would rewrite" if args.dry_run else "Rewrote"
    print(f"{verb} {total} reference(s) across {total_files} file(s).")
    if args.dry_run and total:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
