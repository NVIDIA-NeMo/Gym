# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Togetherbench/SWE-Together, Apache-2.0, revision 891d19eb4b3a64a47c3d49bbd066a311e0133254.

"""Discover and read agent configuration files from the task repository.

Scans the working directory inside the Docker container for well-known
config files (CLAUDE.md, AGENTS.md, .claude/, .ai/, .cursor/, etc.)
and returns their concatenated contents for injection into agent prompts.

This ensures all agents get the same repo context regardless of whether
they natively auto-discover these files (e.g., Claude Code reads CLAUDE.md
automatically, but Terminus2 and Codex do not).
"""

from __future__ import annotations

import logging

from nemo_gym.sandbox.api import AsyncSandbox


log = logging.getLogger(__name__)

# Shell script that finds well-known config files and prints them with
# delimiters so we can parse them back on the Python side.
_DISCOVER_SCRIPT = r"""
files=""
for f in CLAUDE.md AGENTS.md .cursorrules .github/copilot-instructions.md; do
    [ -f "$f" ] && files="$files $f"
done
for d in .claude .ai .cursor; do
    if [ -d "$d" ]; then
        for f in $(find "$d" -maxdepth 2 -type f \( -name '*.md' -o -name 'rules' \) 2>/dev/null); do
            files="$files $f"
        done
    fi
done
for f in $files; do
    echo "===FILE:${f}==="
    head -c 8000 "$f"
    echo ""
    echo "===END==="
done
""".strip()


async def discover_repo_config_files(
    environment: AsyncSandbox,
    max_total_chars: int = 30_000,
) -> str:
    """Scan the repo working directory for agent config files.

    Runs a shell command inside *environment* to find and read files.
    Returns a formatted markdown string suitable for prepending to an
    agent instruction, or an empty string if nothing was found.
    """
    try:
        result = await environment.exec(
            _DISCOVER_SCRIPT,
            timeout_s=15,
        )
    except Exception as exc:
        log.warning("Failed to discover repo config files: %s", exc)
        return ""

    stdout = (result.stdout or "").strip()
    if not stdout:
        log.debug("No repo config files found")
        return ""

    # Parse ===FILE:path=== ... ===END=== blocks
    sections: list[tuple[str, str]] = []
    current_file: str | None = None
    current_lines: list[str] = []

    for line in stdout.split("\n"):
        if line.startswith("===FILE:") and line.endswith("==="):
            if current_file is not None:
                sections.append((current_file, "\n".join(current_lines).strip()))
            current_file = line[8:-3]
            current_lines = []
        elif line == "===END===":
            if current_file is not None:
                sections.append((current_file, "\n".join(current_lines).strip()))
                current_file = None
                current_lines = []
        else:
            current_lines.append(line)

    # Handle unterminated last block
    if current_file is not None:
        sections.append((current_file, "\n".join(current_lines).strip()))

    if not sections:
        return ""

    found_names = [f for f, _ in sections]
    log.info("Found repo config files: %s", found_names)

    # Build formatted output, respecting size limit
    parts = [
        "## Repository Configuration Files\n",
        "The following configuration files were found in the repository. "
        "Follow any project-specific guidelines they contain.\n",
    ]
    total = sum(len(p) for p in parts)

    for filepath, content in sections:
        section = f"### {filepath}\n\n{content}\n"
        if total + len(section) > max_total_chars:
            parts.append(f"### {filepath}\n\n(truncated — file too large)\n")
            break
        parts.append(section)
        total += len(section)

    return "\n".join(parts)
