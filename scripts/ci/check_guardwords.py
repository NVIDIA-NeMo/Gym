# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report locations of private guardwords introduced by a pull request."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _load_patterns(path: Path) -> tuple[list[str], bool]:
    """Load literal patterns without including their values in errors or output."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        matching = document["matching"]
        entries = document["patterns"]
        if matching["mode"] != "literal" or not isinstance(matching["case_sensitive"], bool):
            raise ValueError
        patterns = [entry["pattern"] for entry in entries]
        if not patterns or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise ValueError
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError, ValueError):
        raise ValueError("invalid private guardword configuration") from None
    return patterns, matching["case_sensitive"]


def find_added_matches(diff: str, patterns: list[str], *, case_sensitive: bool) -> list[str]:
    """Return only file and line locations for matching added diff lines."""
    locations: list[str] = []
    seen: set[str] = set()
    current_path: str | None = None
    new_line_number: int | None = None
    in_hunk = False
    search_patterns = patterns if case_sensitive else [pattern.casefold() for pattern in patterns]

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current_path = None
            new_line_number = None
            in_hunk = False
            continue
        if not in_hunk and line.startswith("+++ "):
            path = line[4:]
            current_path = None if path == "/dev/null" else path
            continue
        hunk = _HUNK_HEADER.match(line)
        if hunk:
            new_line_number = int(hunk.group(1))
            in_hunk = True
            continue
        if current_path is None or new_line_number is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added_line = line[1:]
            candidate = added_line if case_sensitive else added_line.casefold()
            if any(pattern in candidate for pattern in search_patterns):
                location = f"{current_path}:{new_line_number}"
                if location not in seen:
                    seen.add(location)
                    locations.append(location)
            new_line_number += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        elif line.startswith(" "):
            new_line_number += 1
    return locations


def _git_diff(base: str, head: str) -> str:
    """Get an uncontextualized diff; discard Git's diagnostics to keep output safe."""
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--no-color", "--unified=0", "--no-prefix", f"{base}...{head}"],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError("could not inspect pull request changes")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()

    try:
        patterns, case_sensitive = _load_patterns(args.patterns)
        locations = find_added_matches(_git_diff(args.base, args.head), patterns, case_sensitive=case_sensitive)
    except (RuntimeError, ValueError):
        return 2

    if locations:
        print("\n".join(locations))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
