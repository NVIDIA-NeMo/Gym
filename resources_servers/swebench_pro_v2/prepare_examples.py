# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the offline tool for the five committed public smoke inputs."""

from pathlib import Path

from benchmarks.swebench.pro_v2 import prepare as pro_v2


EXAMPLE_FPATH = Path(__file__).parent / "data" / "example.jsonl"


def is_prepared_data_current(fpath: str | Path) -> bool:
    """The committed inputs need the same pinned offline tool as the full benchmark."""
    return Path(fpath).resolve() == EXAMPLE_FPATH.resolve() and pro_v2.RIPGREP_FPATH.is_file()


def prepare() -> Path:
    """Keep the committed task inputs and stage their network-free search tool."""
    if not EXAMPLE_FPATH.is_file():
        raise FileNotFoundError(f"Missing committed SWE-bench Pro V2 examples: {EXAMPLE_FPATH}")
    pro_v2.fetch_ripgrep()
    return EXAMPLE_FPATH
