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
"""Rewrite legacy dataset identifiers to the unified ``source:`` block.

``gitlab_identifier:`` / ``huggingface_identifier:`` are deprecated in favour of a single
discriminated ``source:`` block (see ``DatasetConfig``). This tool rewrites existing YAML
configs in place. It is a line-based transform: it only renames the mapping key and injects
the ``type`` discriminator, so all surrounding comments, ordering, and formatting are preserved
byte-for-byte and the diff stays minimal.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


# Matches a block-style legacy identifier key on its own line, e.g. "        gitlab_identifier:".
# Flow style ("gitlab_identifier: {...}") and trailing content are intentionally not matched —
# they are rare here and rewriting them blindly would risk corrupting the file.
_LEGACY_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>gitlab_identifier|huggingface_identifier):\s*$")
_BACKEND_FOR_KEY = {"gitlab_identifier": "gitlab", "huggingface_identifier": "huggingface"}


def migrate_config_text(text: str) -> Tuple[str, int]:
    """Rewrite legacy identifier blocks in ``text`` to ``source:`` blocks.

    Returns the rewritten text and the number of identifier blocks migrated. The existing
    nested fields (``dataset_name``/``version``/``artifact_fpath`` or ``repo_id``) are already
    indented one level under the key, which is exactly where ``source:``'s children belong, so
    they are left untouched.
    """
    migrated = 0
    out_lines: List[str] = []
    for line in text.split("\n"):
        match = _LEGACY_KEY_RE.match(line)
        if match is None:
            out_lines.append(line)
            continue
        indent = match.group("indent")
        backend = _BACKEND_FOR_KEY[match.group("key")]
        out_lines.append(f"{indent}source:")
        out_lines.append(f"{indent}  type: {backend}")
        migrated += 1
    return "\n".join(out_lines), migrated


def migrate_file(path: Path, dry_run: bool = False) -> int:
    """Migrate a single YAML file in place. Returns the number of blocks migrated."""
    original = path.read_text()
    rewritten, migrated = migrate_config_text(original)
    if migrated and not dry_run:
        path.write_text(rewritten)
    return migrated


def _iter_yaml_files(paths: List[Path]):  # pragma: no cover
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.yaml"))
        else:
            yield path


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="YAML files or directories to migrate.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()

    total_files = 0
    total_blocks = 0
    for path in _iter_yaml_files(args.paths):
        migrated = migrate_file(path, dry_run=args.dry_run)
        if migrated:
            total_files += 1
            total_blocks += migrated
            verb = "would migrate" if args.dry_run else "migrated"
            print(f"{verb} {migrated} block(s): {path}")

    verb = "Would migrate" if args.dry_run else "Migrated"
    print(f"{verb} {total_blocks} identifier block(s) across {total_files} file(s).")
    if args.dry_run and total_blocks:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
