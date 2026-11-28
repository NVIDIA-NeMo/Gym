# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the AutomationBench rows for the session harness presets (opencode.yaml, openclaw.yaml, hermes.yaml,
pi.yaml, codex.yaml).

The rows match ``prepare.py``'s but carry no tool list and no output budget, which these harnesses refuse in the
request. They take the tools from the resources server's MCP grant and the output budget from their own config.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.automationbench.prepare import prepare as prepare_rows


def prepare(
    domains: list[str] | None = None,
    size: int = -1,
    max_turns: int = 50,
    out: Path | None = None,
) -> Path:
    """Write the harness-neutral taskset to JSONL and return the path written; see ``prepare.prepare``."""
    return prepare_rows(domains=domains, size=size, max_turns=max_turns, out=out, harness_neutral=True)


if __name__ == "__main__":
    prepare()
