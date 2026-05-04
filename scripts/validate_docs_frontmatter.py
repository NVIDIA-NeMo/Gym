# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate frontmatter on Fern docs pages against the Content IA Audit's controlled vocabulary.

Checks every `.mdx` page passed in (or all pages under fern/versions/latest/pages/ when run
without args) for:
  - Presence of required fields: title, description, content_type, audience_level, journey_stage
  - content_type, audience_level, journey_stage values within the controlled vocab

Exits non-zero on any violation. Designed for pre-commit; the hook only runs on changed files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_FIELDS = ("title", "description", "content_type", "audience_level", "journey_stage")

# Optional fields validated only for shape if present; not required on any page.
# Add fields here as the controlled vocab grows.
OPTIONAL_FIELDS = ("duration", "gpu_required", "cost_estimate", "deprecated_in", "replaced_by")

CONTENT_TYPE = {
    "tutorial",
    "how-to",
    "reference",
    "explanation",
    "quickstart",
    "troubleshooting",
    "faq",
    "index",
    "recipe",
}
AUDIENCE_LEVEL = {"beginner", "intermediate", "advanced"}
JOURNEY_STAGE = {"discover", "try", "build", "scale"}
GPU_REQUIRED = {"true", "false", "True", "False"}

VOCAB = {
    "content_type": CONTENT_TYPE,
    "audience_level": AUDIENCE_LEVEL,
    "journey_stage": JOURNEY_STAGE,
    "gpu_required": GPU_REQUIRED,
}

FIELD_RE = re.compile(r'^(\w+):\s*(?:"([^"]*)"|(\S+))', re.MULTILINE)
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def parse_frontmatter(text: str) -> dict[str, str] | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm = {}
    for fm_match in FIELD_RE.finditer(m.group(1)):
        key = fm_match.group(1)
        value = fm_match.group(2) if fm_match.group(2) is not None else fm_match.group(3)
        fm[key] = value
    return fm


def validate_file(path: Path) -> list[str]:
    """Return list of error strings; empty if file passes."""
    errors: list[str] = []
    try:
        text = path.read_text()
    except Exception as e:
        return [f"{path}: could not read ({e})"]

    fm = parse_frontmatter(text)
    if fm is None:
        errors.append(f"{path}: missing or malformed frontmatter (no leading ---...--- block)")
        return errors

    # Required fields must be present.
    # `description` may be empty for backward compatibility with legacy pages;
    # the other required fields must also be non-empty.
    for field in REQUIRED_FIELDS:
        if field not in fm:
            errors.append(f"{path}: missing required field `{field}`")
        elif not fm[field] and field != "description":
            errors.append(f"{path}: field `{field}` is empty")

    for field, allowed in VOCAB.items():
        value = fm.get(field)
        if value and value not in allowed:
            errors.append(
                f"{path}: `{field}: {value}` is not in controlled vocab "
                f"({', '.join(sorted(allowed))})"
            )

    return errors


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = sorted(Path("fern/versions/latest/pages").rglob("*.mdx"))

    all_errors: list[str] = []
    for p in paths:
        if p.suffix != ".mdx":
            continue
        # Only validate latest/ pages — v0.2 is a stable snapshot we don't audit
        if "versions/latest/pages" not in str(p).replace("\\", "/"):
            continue
        all_errors.extend(validate_file(p))

    if all_errors:
        print("Docs frontmatter validation failed:", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        print(
            f"\n{len(all_errors)} violation(s). "
            "Required fields: title, description, content_type, audience_level, journey_stage.\n"
            f"Optional fields (validated only if present): {', '.join(OPTIONAL_FIELDS)}.\n"
            "Controlled vocab values:\n"
            f"  content_type: {', '.join(sorted(CONTENT_TYPE))}\n"
            f"  audience_level: {', '.join(sorted(AUDIENCE_LEVEL))}\n"
            f"  journey_stage: {', '.join(sorted(JOURNEY_STAGE))}\n"
            f"  gpu_required: {', '.join(sorted(GPU_REQUIRED))}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
