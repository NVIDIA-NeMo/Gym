# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard: the constrained wrapper must never depend on the agentic-if repo.

The dependency direction is one-way by design: the grading core here is the
single source of truth, and agentic-if (the constraint-development repo)
re-exports it via shims. This test keeps the old backward edge — a runtime
import of ``instruction_pool`` from an external checkout — from quietly
returning.

Prose mentions of the agentic-if repo (hyphenated, in comments/docstrings) are
fine; identifiers and env vars that would wire a code path back to it are not.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Identifier-level tokens that indicate a code dependency, not a prose mention.
FORBIDDEN_TOKENS = (
    "instruction_pool",
    "agentic_if",
    "AGENTIC_IF",
)


def test_no_agentic_if_code_references():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path == Path(__file__).resolve() or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for token in FORBIDDEN_TOKENS:
                if token in line:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "swe_agents_constrained must not reference the agentic-if repo in code "
        "(single source of truth lives in grading/; agentic-if depends on Gym, "
        "never the reverse):\n" + "\n".join(offenders)
    )


def test_config_has_no_agentic_if_knobs():
    config = PACKAGE_ROOT / "configs" / "swe_agents_constrained.yaml"
    text = config.read_text(encoding="utf-8")
    for token in ("agentic_if_repo", "AGENTIC_IF_REPO", "../agentic-if"):
        assert token not in text, f"{config.name} still references {token!r}"
