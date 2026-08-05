# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural guards for run_gdpval_rollouts.sh.

`bash -n` accepts a function defined inside another function's body, and the
script only fails at runtime with "command not found" when an earlier caller
reaches it first -- after a Slurm allocation has already been granted. These
tests catch that class of defect without needing a cluster.
"""

import re
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_gdpval_rollouts.sh"

DEF_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{", re.M)


def _definitions() -> list[tuple[str, int, str]]:
    """(name, 1-based line, leading whitespace) for every function definition."""
    text = RUNNER.read_text(encoding="utf-8")
    return [(m.group("name"), text[: m.start()].count("\n") + 1, m.group("indent")) for m in DEF_RE.finditer(text)]


def test_every_function_is_defined_at_top_level() -> None:
    nested = [(n, ln) for n, ln, indent in _definitions() if indent]
    assert not nested, (
        "function(s) defined inside another function body; they only exist once "
        f"the enclosing function has run: {nested}"
    )
