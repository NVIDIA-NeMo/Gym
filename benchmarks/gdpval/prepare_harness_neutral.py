# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the GDPVal rows for harnesses that get the task tools from the resources server.

The rows match ``prepare.py``'s but carry no tool list, which session harnesses such as Pi and Codex reject in the
request. Pi and Codex reach finish, abandon_task_finish, web_search and fetch_web_page through the resources server's
MCP grant. Use these rows with ``prompts/harness-neutral.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.gdpval.prepare import DATA_DIR, write_benchmark


OUTPUT_FPATH = DATA_DIR / "gdpval_benchmark_harness_neutral.jsonl"


def prepare() -> Path:
    return write_benchmark(OUTPUT_FPATH, tools=None)


if __name__ == "__main__":
    prepare()
